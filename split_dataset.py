import os
import random
import shutil

# Configuration
IMAGES_DIR = "all_images"
LABELS_DIR = "all_labels"
OUTPUT_DIR = "dataset"
SPLIT_RATIO = 0.8 # 80% for training, 20% for validation

def setup_dirs():
    dirs = [
        f"{OUTPUT_DIR}/images/train",
        f"{OUTPUT_DIR}/images/val",
        f"{OUTPUT_DIR}/labels/train",
        f"{OUTPUT_DIR}/labels/val"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(LABELS_DIR):
        print(f"Error: Please create an '{IMAGES_DIR}' folder and an '{LABELS_DIR}' folder.")
        print("Put your raw .jpg images in 'all_images' and your .txt files from MakeSense in 'all_labels'.")
        return

    setup_dirs()
    
    # Get all images
    images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print(f"No images found in {IMAGES_DIR} folder.")
        return

    # Shuffle for random distribution
    random.shuffle(images)
    
    train_count = int(len(images) * SPLIT_RATIO)
    
    train_images = images[:train_count]
    val_images = images[train_count:]
    
    print(f"Total images: {len(images)} | Training: {len(train_images)} | Validation: {len(val_images)}")
    
    # Helper to copy files
    def copy_files(file_list, split_type):
        for img_name in file_list:
            # Copy image
            src_img = os.path.join(IMAGES_DIR, img_name)
            dst_img = os.path.join(OUTPUT_DIR, "images", split_type, img_name)
            shutil.copy(src_img, dst_img)
            
            # Copy corresponding label (if it exists)
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            src_txt = os.path.join(LABELS_DIR, txt_name)
            dst_txt = os.path.join(OUTPUT_DIR, "labels", split_type, txt_name)
            
            # If there's no txt file, it's a background image (perfect for ignoring waterfalls!)
            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)

    print("Copying training files...")
    copy_files(train_images, "train")
    
    print("Copying validation files...")
    copy_files(val_images, "val")
    
    print(f"\nDataset successfully built in the '{OUTPUT_DIR}' folder!")
    print("You are now ready to run: python train.py")

if __name__ == "__main__":
    main()
