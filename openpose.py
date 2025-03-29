import sys
import cv2
import os
import json
import numpy as np

# Import OpenPose
sys.path.append('/opt/openpose/build/python')
os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':/opt/openpose/build/python'
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/opt/openpose/build/x64/Release:/opt/openpose/bin'

try:
    import pyopenpose as op
except ImportError as e:
    print("Error: OpenPose library could not be found. Check BUILD_PYTHON in CMake.")
    raise e

    # Set OpenPose parameters
    params = {"model_folder": "/opt/openpose/models/"}

    # Initialize OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Define dataset and output paths
    input_dir = "/opt/openpose/images"
    output_img_dir = "/opt/openpose/output/openpose_img"
    output_json_dir = "/opt/openpose/output/openpose_json"

    # Ensure output directories exist
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)

    # Define body part connections
    body_parts = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), 
                  (17, 15), (15, 0), (0, 16), (16, 18), (1, 8), (8, 9), (9, 10), 
                  (10, 11), (11, 24), (11, 22), (22, 23), (8, 12), (12, 13), 
                  (13, 14), (14, 21), (14, 19), (19, 20)]

    colors = [(255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255),  
              (255, 165, 0), (255, 165, 0), (255, 165, 0),  
              (0, 255, 0), (0, 255, 0), (0, 255, 0),  
              (255, 0, 0), (255, 0, 0), (255, 0, 0),  
              (0, 255, 0), (0, 255, 0), (0, 255, 0)]  

    # Process each image
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(input_dir, filename)
        imageToProcess = cv2.imread(image_path)
        if imageToProcess is None:
            print(f"Skipping {filename}: Failed to load image.")
            continue

        datum = op.Datum()
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints
        face_keypoints = datum.faceKeypoints
        left_hand_keypoints = datum.handKeypoints[0]
        right_hand_keypoints = datum.handKeypoints[1]

        if keypoints is None:
            print(f"Skipping {filename}: No keypoints detected.")
            continue

        # Create a black background image
        h, w, _ = imageToProcess.shape
        output_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw keypoints and connections
        person = keypoints[0]  # First person detected
        valid_connections = []

        for i, (p1, p2) in enumerate(body_parts):
            if p1 >= person.shape[0] or p2 >= person.shape[0]:
                continue
            if person[p1][2] > 0.2 and person[p2][2] > 0.2:
                pt1 = tuple(map(int, person[p1][:2]))
                pt2 = tuple(map(int, person[p2][:2]))
                if pt1 != pt2:
                    valid_connections.append((pt1, pt2, colors[i % len(colors)]))

        # Draw valid connections
        for (pt1, pt2, color) in valid_connections:
            cv2.line(output_image, pt1, pt2, color, 3)
            cv2.circle(output_image, pt1, 5, color, -1)
            cv2.circle(output_image, pt2, 5, color, -1)

        # Save output image
        output_image_path = os.path.join(output_img_dir, f"{os.path.splitext(filename)[0]}_skeleton.jpg")
        cv2.imwrite(output_image_path, output_image)
        print(f"Processed image saved at: {output_image_path}")

        # Format keypoints for JSON
        def format_keypoints(keypoints_array):
            if keypoints_array is None:
                return []
            return [float(coord) for person in keypoints_array for kp in person for coord in kp]

        json_output = {
            "version": 1.3,
            "people": [{
                "person_id": [-1],
                "pose_keypoints_2d": format_keypoints(keypoints),
                "face_keypoints_2d": format_keypoints(face_keypoints),
                "hand_left_keypoints_2d": format_keypoints(left_hand_keypoints),
                "hand_right_keypoints_2d": format_keypoints(right_hand_keypoints),
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }]
        }

        # Save keypoints as JSON
        output_json_path = os.path.join(output_json_dir, f"{os.path.splitext(filename)[0]}_skeleton.json")
        with open(output_json_path, "w") as json_file:
            json.dump(json_output, json_file, indent=4)

        print(f"Keypoints saved at: {output_json_path}")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(-1)
