import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from ultralytics import YOLO
import random
import cv2
import torch
from PIL import Image 
import sys
import os
from contextlib import redirect_stdout
import logging

# Get the logger used by YOLO and adjust its level
yolo_logger = logging.getLogger('ultralytics')
yolo_logger.setLevel(logging.WARNING)


def get_random_image_path(image_dir):
    """
    Selects a random image from the specified directory. If there are class subfolders,
    it returns a random class and image. If there are no subfolders, returns '_' and the image path.

    Args:
    image_dir (str): The path to the directory.

    Returns:
    tuple: A tuple containing the class/subfolder name (or '_') and the full path to the random image.
    """
    # Check if there are any directories in the specified path
    class_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

    if class_dirs:
        # If there are class subfolders, proceed as before
        chosen_class = random.choice(class_dirs)
        class_dir_path = os.path.join(image_dir, chosen_class)
        all_images = os.listdir(class_dir_path)
    else:
        # If there are no subfolders, treat the entire directory as containing images
        chosen_class = '_'
        class_dir_path = image_dir
        all_images = [file for file in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, file))]

    # Select a random image from the chosen directory
    image_name = random.choice(all_images)
    # Construct the full path to the image
    image_path = os.path.join(class_dir_path, image_name)

    return chosen_class, image_path


def detect_and_classify(image):
    # Perform image classification
    cls_results = model_cls.predict(image)
    for result in cls_results: 
        # classified_pathologies = result.names[result.probs.top1]
        classified_pathologies = [result.names[patho] for patho in result.probs.top5]
        print(f"Top 1 Thoracic Pathologies Identified: {result.names[result.probs.top1]}")
        print(f'Confidence Score for Top 1 Classifier: {round(max(result.probs.top5conf).item()*100, 2)}%')
        print('')
        print(f"Top 1 Thoracic Pathologies Identified: {classified_pathologies}")
        print("Confidence Scores for Top 5 Classifiers: " + ", ".join(f"{score.item()*100:.2f}%" for score in result.probs.top5conf))

        
    # Perform object detection
    det_results = model_det(image)
    for result in det_results:
        yolo_plot_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
        plt.imshow(yolo_plot_image)
        plt.axis('off')  # This removes the axes with ticks and labels
        
        print('')
        
        if result.boxes.cls.nelement() == 0:
            # print("No Pathology Detected")
            classes = "Object Detection: No Pathology Detected"
            print(classes)
        else:
            # classes = f"Pathology Detected: {result.names[result.boxes.cls.cpu().item()]}"
            detected_classes_indices = result.boxes.cls.cpu().tolist()  # This will work for 1 or more elements
            detected_class_names = [result.names[idx] for idx in detected_classes_indices]  # Map indices to names

            # Join class names into a single string if there are multiple detections
            classes_str = ", ".join(detected_class_names)
            classes = f"Pathology Detected: {classes_str}"
            print(f"Thoracis Pathologies Detected: {classes}")
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(yolo_plot_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.imshow('image', yolo_plot_image)
    plt.show()
    return yolo_plot_image, classes




if __name__ == '__main__':
    
    classify_model = r"best_classify.pt"
    det_model = r"best_detect.pt"
    # image_dir = r"C:/Users/pault/Documents/5. Projects/8. NIH Chest X-rays/data/classification_YOLO/train"
    root_dir = os.getcwd()
    image_dir = "sample_images"
    
    class_name, image_path = get_random_image_path(os.path.join(root_dir, image_dir))

    print(f"Selected Image Class: {class_name}")
    print('')
    print('MACHINE LEARNING MODEL')

    model_cls = YOLO(classify_model, task = 'classify')
    model_det = YOLO(det_model, task = 'detect')


    yolo_plot_image, classes = detect_and_classify(image_path) 
