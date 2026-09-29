from ultralytics import YOLO

def main():
    print("Initializing YOLOv8 model for training...")
    # Initialize from the small pre-trained model (fast and accurate)
    model = YOLO('yolov8s.pt') 

    print("Starting training on your custom dataset...")
    
    # Train the model
    results = model.train(
        data='dataset.yaml',       
        epochs=10,                 # Reduced to 10 for CPU training (use Google Colab for full 50-epoch run)
        imgsz=640,                 # Standard resolution
        batch=8,                   # Lowered to 8 to prevent Out-Of-Memory errors on your GPU
        device='cpu',              # Using CPU (no CUDA GPU detected on this machine)
        workers=4,
        
        # Specific tweaks for dense material
        mosaic=1.0,                
        mixup=0.2,                 
        copy_paste=0.1,
        
        optimizer='auto',
        lr0=0.01,
        patience=20                
    )
    
    print("\n" + "="*50)
    print("Training complete! Your custom model is saved at:")
    print("runs/detect/train/weights/best.pt")
    print("="*50)

if __name__ == "__main__":
    main()
