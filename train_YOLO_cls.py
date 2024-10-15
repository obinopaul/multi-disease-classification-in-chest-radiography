import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import glob
import os

def main():
    root_dir = r"C:\Users\pault\Documents\5. Projects\8. NIH Chest X-rays"
    os.chdir(root_dir)

    data_path = "data\classification_YOLO"
    # Verify the path to the pretrained weights and configuration
    model_path = 'yolov8n-cls.pt'  # Adjust this to your specific model file

    # Initialize the model
    model = YOLO(model_path, task='classify')  # Use task='train' if the intent is to train
    model.to('cuda')

    # Training the model, adjust parameters as necessary
    results = model.train(data=data_path, epochs=100, imgsz=640)

    # Display training results
    print("Training completed with the following results:")
    print(results)

    # Save the trained model weights if needed
    output_model_path = 'trained_weights_cls.pt'
    model.save(output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == '__main__':
    main()
