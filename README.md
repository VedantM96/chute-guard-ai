# ChuteGuard AI 🚧📹

**ChuteGuard AI** is a hardened, real-time Computer Vision module designed to detect structural material buildup and prevent critical choke-points in industrial conveyor chutes.

Rather than relying on naive bounding boxes, this system utilizes a custom-trained **YOLOv8** model paired with an advanced **Hybrid Contour Mapping** pipeline. This ensures the system tracks the precise physical geometry of stationary blockages while completely ignoring the chaotic movement of free-falling material, water, and debris.

## ✨ Key Features

* **Custom YOLOv8 Inference:** Fine-tuned object detection specifically for industrial ore and material buildup.
* **Hybrid Contour Mapping:** Dynamically thresholds pixels *inside* the AI's bounding box to "shrink-wrap" the detection mask to the exact, irregular shape of the material pile.
* **Precision Filtering Pipeline:** 
  * *Morphological Cleanup:* Erases noise and speckles.
  * *Aspect Ratio Gating:* Rejects vertically thin streaks (falling water/ore).
  * *Floor-Anchored Checks:* Validates that buildup is resting on the chute floor rather than floating in mid-air.
* **Temporal Debouncing (EMA):** Utilizes an Exponential Moving Average and frame-persistence queue to prevent single-frame anomalies (like a splash or camera glare) from triggering false alarms.
* **Real-Time Video Pacing:** Built-in sync engine forces playback to match the camera's true FPS. Automatically drops inference frames if the CPU falls behind, ensuring the blockage timeline stays perfectly true to real-world time.

## 🛠️ Tech Stack
* **Python 3.x**
* **Ultralytics YOLOv8** (Object Detection)
* **OpenCV (cv2)** (Image processing, contour masking, UI)
* **NumPy** (Mask aggregation and mathematical filtering)

## 📂 Project Structure

* `hardened_blockage_analyzer.py`: The core MVP script. Handles live inference, precision filtering, and UI rendering.
* `train.py`: Standard YOLOv8 training script with specific optimizations (mosaic, mixup) for dense industrial materials.
* `extract_frames.py`: Utility to sample raw video into training frames.
* `convert_coco_to_yolo.py`: Utility to convert polygon annotations from MakeSense.ai into YOLO-compatible text files.
* `split_dataset.py`: Automatically shuffles and sorts images/labels into an 80/20 train/validation split.

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install ultralytics opencv-python numpy
   ```

2. **Run the Analyzer:**
   ```bash
   python hardened_blockage_analyzer.py --source "path/to/your/video.mp4" --debug
   ```

3. **Interact with the UI:**
   * **Material Thresh Slider (Top Left):** Adjust this trackbar in real-time to perfectly tune the pixel-brightness contour mapping to your specific lighting conditions.

## 🎛️ Tuning the Pipeline

The system is designed to be highly modular. You can adjust the strictness of the AI by editing the `# --- PRECISION TUNABLES ---` block at the top of `hardened_blockage_analyzer.py`:

```python
REALTIME_SYNC   = True   # Pace playback to the video's real speed
PLAYBACK_SPEED  = 0.5    # 1.0 = real time, 0.5 = half speed (slow motion)
MIN_BLOB_AREA_FRAC   = 0.02    # Blobs must cover >= 2% of the ROI area
MAX_ASPECT_RATIO     = 8.0     # Rejects vertically thin strips (falling material)
```
