import cv2
import numpy as np
import os
from rembg import remove
from PIL import Image

def remove_background(image_path):
    input_image = Image.open(image_path)
    return remove(input_image)

def make_foreground_white(image):
    alpha_channel = np.array(image)[:, :, 3]  # Extract the alpha channel
    white_foreground = np.full_like(np.array(image)[:, :, :3], (255, 255, 255), dtype=np.uint8)
    return np.dstack((white_foreground, alpha_channel))

def make_black_background(image):
    alpha_channel = image[:, :, 3]  # Extract the alpha channel
    rgb_channels = image[:, :, :3]  # Extract RGB channels
    black_background = np.zeros_like(rgb_channels, dtype=np.uint8)
    alpha_factor = alpha_channel[:, :, None] / 255.0
    return (rgb_channels * alpha_factor + black_background * (1 - alpha_factor)).astype(np.uint8)

def process_image(image_path, output_path):
    bg_removed = remove_background(image_path)
    white_foreground = make_foreground_white(bg_removed)
    final_image = make_black_background(white_foreground)
    cv2.imwrite(output_path, final_image)
    print(f"Processed image saved at: {output_path}")

folder_dir = "/Users/nandanpatel/Projects/AI Clothing Tryon/Dataset/train/cloth"
output_folder = "/Users/nandanpatel/Projects/AI Clothing Tryon/Dataset/train/cloth_mask"
os.makedirs(output_folder, exist_ok=True)
for image in os.listdir(folder_dir):
    if (image.endswith(".jpg")):
        input_image = os.path.join(folder_dir, image)
        output_image = os.path.join(output_folder, os.path.basename(image))
        process_image(input_image, output_image)
        print("Saved as ", os.path.basename(image))