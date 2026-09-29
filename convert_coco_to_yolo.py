import json
import os

# Configuration
COCO_JSON_FILE = "labels.json"  # The file you will download from MakeSense
OUTPUT_LABELS_DIR = "all_labels"

def convert_coco_to_yolo():
    if not os.path.exists(COCO_JSON_FILE):
        print(f"Error: Could not find '{COCO_JSON_FILE}'.")
        print("Please put the file in this folder and make sure it is named exactly 'labels.json'.")
        return

    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    with open(COCO_JSON_FILE, 'r') as f:
        coco_data = json.load(f)

    # Map image IDs to their filenames and dimensions
    images_info = {}
    for img in coco_data.get('images', []):
        images_info[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        }

    # Process annotations
    annotations_count = 0
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in images_info:
            continue

        img_info = images_info[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # COCO bbox format is [top_left_x, top_left_y, width, height]
        bbox = ann['bbox']
        x, y, w, h = bbox
        
        # YOLO format is [center_x, center_y, width, height] (Normalized 0.0 to 1.0)
        center_x = (x + w / 2.0) / img_width
        center_y = (y + h / 2.0) / img_height
        yolo_w = w / img_width
        yolo_h = h / img_height
        
        # category_id (MakeSense usually starts at 0 or 1, we want 0 for YOLO)
        # We will assume class index 0 since we only have one class: wall_buildup
        class_id = 0 
        
        # Write to txt file
        txt_filename = os.path.splitext(img_info['file_name'])[0] + ".txt"
        txt_path = os.path.join(OUTPUT_LABELS_DIR, txt_filename)
        
        with open(txt_path, 'a') as txt_file:
            txt_file.write(f"{class_id} {center_x:.6f} {center_y:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")
            
        annotations_count += 1

    print(f"Success! Converted {annotations_count} annotations into YOLO format inside the '{OUTPUT_LABELS_DIR}' folder.")

if __name__ == "__main__":
    convert_coco_to_yolo()
