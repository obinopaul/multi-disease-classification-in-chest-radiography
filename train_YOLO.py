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

    # Verify the path to the pretrained weights and configuration
    model_path = 'yolov8n.pt'  # Adjust this to your specific model file
    data_path = r'data\detections\yolo\data.yaml'  # Ensure this path is correct

    # Check if data configuration file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data config file {data_path} does not exist.")

    # Initialize the model
    model = YOLO(model_path, task='detect') 
    model.to('cuda')

    # Training the model, adjust parameters as necessary
    results = model.train(data=data_path, epochs=100, imgsz=640)

    # Display training results
    print("Training completed with the following results:")
    print(results)

    # Optionally, plot the results if the framework supports it
    if hasattr(results, 'plot'):
        results.plot()  # or use a custom plotting function if available

    # Save the trained model weights if needed
    output_model_path = 'trained_weights.pt'
    model.save(output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == '__main__':
    main()
