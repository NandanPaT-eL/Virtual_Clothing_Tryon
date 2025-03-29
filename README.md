# Virtual Clothin TryOn
# Dataset Generation

## Overview
This guide provides details on generating the dataset structure required for AI Clothing Try-On. The dataset consists of `train/` and `test/` directories, each containing `cloth/` and `image/` subdirectories. Additional processing generates `cloth_mask/`, `image-parse-v3/`, `openpose_img/`, and `openpose_json/` directories.

## Requirements
### Hardware
- **GPU**: NVIDIA GPU with CUDA support (Recommended: NVIDIA RTX series)
- **Memory**: At least 8GB RAM

### Software
#### Operating System
- Windows 10/11 (Recommended)
- Ubuntu (Optional)

#### Dependencies
- Python 3.8+
- OpenCV (cv2)
- numpy
- Pillow (PIL)
- rembg
- Docker
- CUDA
- MS Visual Studio (For Windows)

## Installation
### Step 1: Set Up Virtual Environment
```sh
python -m venv dataset_env
source dataset_env/bin/activate  # On Windows use: dataset_env\Scripts\activate
```

### Step 2: Install Dependencies
```sh
pip install numpy pillow opencv-python rembg
```

### Step 3: Install CUDA and cuDNN
#### **For Windows:**
1. **Download CUDA Toolkit:**  
   - Go to [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select your operating system and download the latest stable version.
   - Install it and ensure to add CUDA to your system’s `PATH`.
   
2. **Download cuDNN:**  
   - Visit [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) and log in to download.
   - Extract and copy the contents to your CUDA installation directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\`)

#### **For Linux:**
```sh
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
```

### Step 4: Install Microsoft Visual Studio (Windows Users Only)
1. **Download Visual Studio:**
   - Visit [Microsoft Visual Studio](https://visualstudio.microsoft.com/)
   - Download and install the **Community Edition**.
2. **Enable C++ Development Tools:**
   - During installation, select **Desktop Development with C++**.
3. **Set Up Environment Variables:**
   - Open **System Properties > Advanced > Environment Variables**
   - Add `C:\Program Files (x86)\Microsoft Visual Studio\<version>\Community\VC\Auxiliary\Build` to `PATH`.

## Dataset Structure
The dataset should be organized as follows:
```
dataset/
│-- train/
│   │-- cloth/  # Original cloth images
│   │-- cloth_mask/  # Generated cloth masks
│   │-- image/  # Original images
│   │-- image-parse-v3/  # Processed images (output of SCHP)
│   │-- openpose_img/  # OpenPose processed images
│   │-- openpose_json/  # OpenPose keypoint JSON files
│
│-- test/
│   │-- cloth/  # Original cloth images
│   │-- cloth_mask/  # Generated cloth masks
│   │-- image/  # Original images
│   │-- image-parse-v3/  # Processed images (output of SCHP)
│   │-- openpose_img/  # OpenPose processed images
│   │-- openpose_json/  # OpenPose keypoint JSON files
```

## 1. Generating Cloth Masks
To generate the cloth masks from the `cloth/` directory, run the following command:
```sh
python cloth_mask.py
```

### Cloth Mask Generation Process
The script `cloth_mask.py` follows these steps:
1. Reads images from the `cloth/` directory.
2. Removes the background using `rembg`.
3. Converts the foreground to white and background to black.
4. Saves the processed mask images in the `cloth_mask/` directory.

## 2. OpenPose Integration
To generate OpenPose images and keypoints:
```sh
python openpose.py
```

### OpenPose Processing Steps
1. Uses OpenPose to extract keypoints from images in `image/`.
2. Saves the skeleton images in `openpose_img/`.
3. Saves the keypoints JSON files in `openpose_json/`.

### Docker Image Setup
The OpenPose processing runs inside a Docker container. The package `virtual_clothing_tryon openpose_cmu` contains the necessary setup. Ensure you have CUDA and MS Visual Studio installed before using OpenPose.

To build and run the OpenPose container:
```sh
docker build -t openpose_cmu .
docker run --gpus all -v /path/to/dataset:/opt/openpose -it openpose_cmu
```

## 3. Human Parsing with SCHP
To perform semantic human parsing, run the following command:
```sh
python image_parse.py
```

### SCHP Processing Steps
1. Loads pre-trained SCHP model.
2. Parses human images from `image/`.
3. Saves parsed outputs in `image-parse-v3/`.

#### **SCHP Repository:**
This implementation is based on the [Self-Correction-Human-Parsing (SCHP)](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) repository.

## Troubleshooting
- **Missing Dependencies**: Ensure all required libraries are installed.
- **Incorrect Image Paths**: Check if the dataset is structured correctly before running scripts.
- **CUDA Not Recognized**: Ensure CUDA is installed and correctly added to the system `PATH`.
- **Docker Issues**: Verify that Docker, CUDA, and MSVS are properly installed.

## Acknowledgements
This project is part of the AI Clothing Try-On dataset preparation pipeline.


