import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video for YOLO training")
    parser.add_argument("--video", type=str, required=True, help="Path to your video file")
    parser.add_argument("--interval", type=int, default=30, help="Extract 1 frame every X frames (e.g., 30 = 1 frame per sec at 30fps)")
    parser.add_argument("--output", type=str, default="dataset/images", help="Output folder for images")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
        
    frame_count = 0
    saved_count = 0
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    
    print(f"Starting extraction... Saving 1 frame every {args.interval} frames.")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        if frame_count % args.interval == 0:
            filename = os.path.join(args.output, f"{video_name}_frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"Saved: {filename}")
            
        frame_count += 1
        
    cap.release()
    print(f"Done! Extracted {saved_count} images to '{args.output}'")

if __name__ == "__main__":
    main()
