import cv2
import numpy as np
import argparse
import os
import time
from collections import deque

def parse_args():
    parser = argparse.ArgumentParser(description="ChuteGuard - Hardened CV Module with Hybrid Contours")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence floor")
    parser.add_argument("--iou", type=float, default=0.40, help="NMS IoU threshold")
    parser.add_argument("--motion_thresh", type=float, default=0.20, help="Max allowed percent of motion in a box before rejecting")
    parser.add_argument("--debounce_frames", type=int, default=15, help="Rolling window size for temporal smoothing")
    parser.add_argument("--debug", action="store_true", help="Show raw boxes and motion mask side-by-side")
    return parser.parse_args()

SHOW_DETECTION_BOXES = False
SHOW_MOTION_MASK = False
ROI_TOP_FRACTION = 0.5

# --- PRECISION TUNABLES ---
REALTIME_SYNC   = True   # pace playback to the video's real speed
PLAYBACK_SPEED  = 0.5    # 1.0 = real time, 0.5 = half speed (slow motion)
MAX_FRAME_SKIP  = 5      # max frames dropped per iteration when lagging
DEBUG_SYNC      = False  # print sync stats to console only
MIN_BLOB_AREA_FRAC   = 0.02    # blob must cover >= 2% of ROI area
MIN_BLOB_WIDTH_FRAC  = 0.10    # and be >= 10% of ROI width
MIN_BLOB_HEIGHT_FRAC = 0.06    # and be >= 6% of ROI height
MIN_SOLIDITY         = 0.55    # contour area / convex hull area
MAX_ASPECT_RATIO     = 8.0     # reject long thin strips (w/h or h/w)
PERSIST_FRAMES       = 15      # blob must be present this many frames
SCORE_EMA_ALPHA      = 0.2     # smoothing of the displayed score
EDGE_MARGIN_PX       = 6       # ignore blobs hugging the ROI border only

def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at '{args.model}'")
        return

    from ultralytics import YOLO
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)

    success, first_frame = cap.read()
    if not success: 
        print(f"\n[ERROR] Could not read the video file at: {args.source}")
        print("Please check if the file path is correct and the video is not corrupted.\n")
        return

    h, w = first_frame.shape[:2]
    
    # --- Auto ROI Setup (Lower 50%) ---
    roi_top = int(h * ROI_TOP_FRACTION)
    ROI_PTS = np.array([[0, roi_top], [w, roi_top], [w, h], [0, h]], dtype=np.int32).reshape((-1, 1, 2))
    
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [ROI_PTS], 255)
    total_roi_area = cv2.countNonZero(roi_mask)

    # --- Hardened Pipeline State ---
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    severity_history = deque(maxlen=args.debounce_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    blob_tracker = []
    smoothed_score = 0.0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps): fps = 30.0
    start_time = time.perf_counter()
    frame_idx = 0

    print("Starting Hardened Pipeline with Hybrid Contours...")
    
    cv2.namedWindow("ChuteGuard Hardened MVP")
    cv2.createTrackbar("Material Thresh", "ChuteGuard Hardened MVP", 90, 255, lambda x: None)
    
    while cap.isOpened():
        loop_start = time.perf_counter()
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            start_time = time.perf_counter()
            frame_idx = 0
            severity_history.clear()
            smoothed_score = 0.0
            continue
            
        frame_idx += 1

        # 1. Motion Detection (Frame Differencing)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        prev_gray_blurred = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        
        frame_diff = cv2.absdiff(prev_gray_blurred, gray_blurred)
        _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        prev_gray = gray.copy() # Update for next frame

        # 2. Raw Inference
        results = model(frame, conf=args.conf, iou=args.iou, verbose=False)
        
        filtered_boxes = []
        filtered_confs = []
        detection_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get threshold from slider for contour mapping
        mat_thresh = cv2.getTrackbarPos("Material Thresh", "ChuteGuard Hardened MVP")

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            # 3. Motion Filtering & Pixel Masking
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Boundary safety check
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                box_area = (x2 - x1) * (y2 - y1)
                
                if box_area == 0: continue

                # Motion check
                box_motion = motion_mask[y1:y2, x1:x2]
                motion_pixels = cv2.countNonZero(box_motion)
                motion_ratio = motion_pixels / box_area

                # Only keep static boxes
                if motion_ratio < args.motion_thresh:
                    filtered_boxes.append(box)
                    filtered_confs.append(conf)
                    
                    # === NEW HYBRID CONTOUR MAPPING ===
                    # Instead of filling a solid rectangle, we threshold the pixels INSIDE the bounding box
                    box_gray = gray_blurred[y1:y2, x1:x2]
                    _, pixel_mask = cv2.threshold(box_gray, mat_thresh, 255, cv2.THRESH_BINARY)
                    
                    # Add only the bright (material) pixels to the detection mask
                    detection_mask[y1:y2, x1:x2] = cv2.bitwise_or(detection_mask[y1:y2, x1:x2], pixel_mask)
                    # ==================================
                    
                    if SHOW_DETECTION_BOXES:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif args.debug and SHOW_DETECTION_BOXES:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)

        # 4. Severity Scoring Formula
        blocked_mask = cv2.bitwise_and(roi_mask, detection_mask)
        
        # --- PRECISION FILTERS ---
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cleaned_mask = cv2.morphologyEx(blocked_mask, cv2.MORPH_OPEN, kernel_open)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
        
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(blocked_mask)
        roi_h = h - int(h * ROI_TOP_FRACTION)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter 1: Size (Significant buildup only)
            if area < MIN_BLOB_AREA_FRAC * total_roi_area: continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < MIN_BLOB_WIDTH_FRAC * w or bh < MIN_BLOB_HEIGHT_FRAC * roi_h: continue
            
            # Filter 2: Shape (Reject vertically thin falling streaks, but allow horizontal floor piles)
            if (bh / max(1, bw)) > MAX_ASPECT_RATIO: continue
            
            # Filter 3: Side walls (Only ignore small noise slivers on the walls, allow massive piles touching walls)
            if (x < EDGE_MARGIN_PX or (x + bw) > w - EDGE_MARGIN_PX) and bw < (MIN_BLOB_WIDTH_FRAC * w * 1.5): 
                continue
                
            # Filter 4: Floor-anchored (Must touch the bottom 15% of the frame)
            if (y + bh) < h - (0.15 * roi_h): continue
            
            # If it survives all filters, draw it to the final mask
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            
        blocked_mask = final_mask
        # --- END PRECISION FILTERS ---

        coverage_ratio = cv2.countNonZero(blocked_mask) / total_roi_area if total_roi_area > 0 else 0

        if coverage_ratio > 0 and len(filtered_confs) > 0:
            mean_conf = sum(filtered_confs) / len(filtered_confs)
            frame_score = (0.6 * coverage_ratio) + (0.4 * mean_conf)
        else:
            frame_score = 0.0

        # 5. Temporal Smoothing (Debounce)
        severity_history.append(frame_score)
        final_severity = sum(severity_history) / len(severity_history)
        final_pct = final_severity * 100.0

        # --- Visualization & UI ---
        red_overlay = np.zeros_like(frame)
        red_overlay[blocked_mask == 255] = [0, 0, 255]
        cv2.addWeighted(red_overlay, 0.45, frame, 1.0, 0, frame)
        cv2.polylines(frame, [ROI_PTS], isClosed=True, color=(255, 255, 0), thickness=2)

        # UI Status Banner
        bar_bg = np.zeros((50, w, 3), dtype=np.uint8)
        if final_severity > 0.50:
            color, status = (0, 0, 255), "CRITICAL BLOCKAGE"
        elif final_severity > 0.20:
            color, status = (0, 165, 255), "WARNING"
        else:
            color, status = (0, 200, 0), "NORMAL"

        fill_width = int(final_severity * w)
        cv2.rectangle(bar_bg, (0, 0), (fill_width, 50), color, -1)
        cv2.putText(bar_bg, f"{status} | Score: {final_pct:.1f}%", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        output = np.vstack([frame, bar_bg])

        if args.debug and SHOW_MOTION_MASK:
            motion_colored = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            motion_small = cv2.resize(motion_colored, (w//3, h//3))
            output[10:10+(h//3), -10-(w//3):-10] = motion_small
            cv2.putText(output, "Motion Filter Mask", (w - (w//3) - 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("ChuteGuard Hardened MVP", output)
        
        proc_time = time.perf_counter() - loop_start
        proc_fps = 1.0 / proc_time if proc_time > 0 else 0
        
        skipped = 0
        if REALTIME_SYNC:
            elapsed = time.perf_counter() - start_time
            target_time = frame_idx / (fps * PLAYBACK_SPEED)
            lag = elapsed - target_time
            
            if lag < 0:
                wait_ms = max(1, int(-lag * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF
                
            if lag > 0:
                while (time.perf_counter() - start_time) > (frame_idx / (fps * PLAYBACK_SPEED)) and skipped < MAX_FRAME_SKIP:
                    if cap.grab():
                        frame_idx += 1
                        skipped += 1
                        severity_history.append(frame_score)
                    else:
                        break
                        
            if DEBUG_SYNC:
                print(f"Proc FPS: {proc_fps:.1f} | Target: {fps:.1f} | Lag: {max(0, lag*1000):.0f}ms | Skipped: {skipped}")
        else:
            key = cv2.waitKey(1) & 0xFF
            
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
