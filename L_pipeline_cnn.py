# Imports
from torchvision import datasets, models, transforms
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from PIL import Image


# Function to do the predicion of up pose
def predict_up(filepath):

    # Load Models
    model_up = models.resnet18(pretrained=True)      

    # Freeze all the layers
    for param in model_up.parameters():
        param.requires_grad = False


    # Modify the final layer to match the number of output classes
    num_classes = 2
    model_up.fc = torch.nn.Linear(model_up.fc.in_features, num_classes)

    model_up_path = 'models/resnet18_aug_up.pth'

    model_state_dict = torch.load(model_up_path)
    model_up.load_state_dict(model_state_dict)


    # Transforms
    train_transforms_up = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49097234, 0.49097234, 0.49097234], std=[0.29159072, 0.29159072, 0.29159072])
    ])

    # Add landmarks
    # Define the pose landmarking model
    model_path = 'pose_landmarker_heavy.task'

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)

    with PoseLandmarker.create_from_options(options) as landmarker:
        # Read the image using OpenCV
        image = cv2.imread(filepath)

        # Perform pose landmarking on the image
        image = mp.Image.create_from_file(filepath)
        pose_landmarker_result = landmarker.detect(image)

        pose_landmarks_list = pose_landmarker_result.pose_landmarks

        # Check if pose landmarks are detected
        if pose_landmarks_list is not None:
            annotated_image = np.copy(image.numpy_view())
            # Loop through the detected poses to visualize.
            for idx in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[idx]

                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])

                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style())

    cv2.imshow('Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Convert the numpy array to a PIL image
    annotated_image_pil = Image.fromarray(annotated_image)

    # Preprocess image
    img = train_transforms_up(annotated_image_pil).unsqueeze(0)
    
    # Make sure the model is in eval mode
    model_up.eval()
    
    # Disable gradients (we don't need them for making predictions)
    with torch.no_grad():
        output = model_up(img)

    # Get the predicted class
    _, prediction = torch.max(output, 1)
    
    # Return prediction
    if prediction.item() == 0:
        return 'correct_seq'
    else:
        return 'wrong_seq'

def predict_down(filepath):

    # Load Models
    model_down = models.resnet18(pretrained=True)  

        # Freeze all the layers
    for param in model_down.parameters():
        param.requires_grad = False

    # Modify the final layer to match the number of output classes
    num_classes = 2
    model_down.fc = torch.nn.Linear(model_down.fc.in_features, num_classes)

    model_down_path = 'models/resnet18_aug_down.pth'

    model_state_dict2 = torch.load(model_down_path)
    model_down.load_state_dict(model_state_dict2)

    train_transforms_down = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49058145, 0.49058145, 0.49058145], std=[0.3018868, 0.3018868, 0.3018868])
    ])


    # Add landmarks
    # Define the pose landmarking model
    model_path = 'pose_landmarker_heavy.task'

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)

    with PoseLandmarker.create_from_options(options) as landmarker:
        # Read the image using OpenCV
        image = cv2.imread(filepath)

        # Perform pose landmarking on the image
        image = mp.Image.create_from_file(filepath)
        pose_landmarker_result = landmarker.detect(image)

        pose_landmarks_list = pose_landmarker_result.pose_landmarks

        # Check if pose landmarks are detected
        if pose_landmarks_list is not None:
            annotated_image = np.copy(image.numpy_view())
            # Loop through the detected poses to visualize.
            for idx in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[idx]

                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])

                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style())

    cv2.imshow('Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Convert the numpy array to a PIL image
    annotated_image_pil = Image.fromarray(annotated_image)

    # Preprocess image
    img = train_transforms_down(annotated_image_pil).unsqueeze(0)
    
    # Make sure the model is in eval mode
    model_down.eval()
    
    # Disable gradients (we don't need them for making predictions)
    with torch.no_grad():
        output = model_down(img)

    # Get the predicted class
    _, prediction = torch.max(output, 1)
    
    # Return prediction
    if prediction.item() == 0:
        return 'correct_seq'
    else:
        return 'wrong_seq'

def main():

    image_path = 'images_with_landmarks/train/up/correct_seq/cor_up_1_16.jpg' # correct 
    image_path = 'images_with_landmarks/test/up/wrong_seq/7_inc.jpg' # wrong

    result = predict_up(image_path)
    print(result)


if __name__ == "__main__":
    main()