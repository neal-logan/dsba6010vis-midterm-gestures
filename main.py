import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn

def grayscale(image):
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
    cv2.COLOR_GRAY2BGR)

def sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
    [0.349, 0.686, 0.168],
    [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(image, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return sepia_img

def blur(image, ksize=15):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Function to invert colors
def invert_colors(image):
    return cv2.bitwise_not(image)


def main():

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Configure MediaPipe Hands with default settings
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    num_landmarks = 21

    # Initialize the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution to HD (1280x720) for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("MediaPipe Hands Capture App")
    print("---------------------------")
    print("Press 'c' to capture & save image and landmarks")
    print("Press 's' to save all captured landmarks to CSV - this will reset captures")
    print("Press 't' to change gesture - this will reset captures")
    print("Press 'f' to toggle between toggling filters with gestures and turning all filters off")
    print("Press 'q' to quit")
    
    # Gesture_map
    gesture_map = {
        'open_palm' : 'no_filter', # Hold your hand flat with fingers spread out and palm facing the camera
        'closed_fist' : 'grayscale', # Close all fingers into a fist
        'peace_sign' : 'sepia', # Extend only your index and middle fingers in a V shape
        'thumbs_up' : 'blur', # Close your fist with only your thumb extended upward
        'pointing' : 'edge_detection', # Extend only your index finger while curling other fingers
        'ok_sign' : 'invert_colors', # Form a circle by touching your thumb and index finger, with other fingers extended
        'other_gesture' : 'no_action' # don't change anything
        }
    
    gesture_list = [
        'open_palm', 
        'closed_fist', 
        'peace_sign', 
        'thumbs_up', 
        'pointing', 
        'ok_sign',
        'other_gesture'
    ]

    # App mode - applying filters with gesture recognition or capturing data
    mode = 'gesture_recognition'  # Change to 'data_capture' for data collection mode

    # Filter application
    # Load pytorch model 'gesture_recognition_model.pth'
    gesture_model = torch.load('gesture_recognition_model.pth', map_location=torch.device('cpu'), weights_only=False) 
    gesture_model.eval()
    for param in gesture_model.parameters():
        param.requires_grad = False
    # Run model on cpu
    device = torch.device('cpu')
    gesture_model.to(device)
    
    
    current_filter = 'no_filter'
    last_filter_change_time = datetime.now()
    filter_change_interval = 5  # seconds
    

    # Data collection application
    save_dir = "captures/"
    capturing_gesture = 'open_palm'
    capture_number = 0

    # Create landmarks header
    landmarks_header = [
        'capturing_gesture',
        'capture_number', 
        'handedness', 
        'score']
    for landmark_id in range(num_landmarks):
        for point_id in ['x', 'y', 'z']:
            landmarks_header.append(f'{landmark_id}_{point_id}')
    
    landmarks_df = pd.DataFrame(columns=landmarks_header)





    # Main loop to process video frames
    while cap.isOpened():
        # Read frame from webcam
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
        
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR image to RGB (CV2 uses BGR but MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        landmarks = hands.process(rgb_frame)
        
        # Create a copy of the frame to draw on
        display_frame = frame.copy()
        
        # Apply the current filter to the frame
        if current_filter == 'grayscale':
            display_frame = grayscale(display_frame)
        elif current_filter == 'sepia':
            display_frame = sepia(display_frame)
        elif current_filter == 'blur':
            display_frame = blur(display_frame)
        elif current_filter == 'edge_detection':
            display_frame = edge_detection(display_frame)
        elif current_filter == 'invert_colors':
            display_frame = invert_colors(display_frame)

        # Draw hand landmarks if detected
        if landmarks.multi_hand_landmarks:
            for hand_landmarks in landmarks.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Show live feed with mode and key information indicated
        if mode == 'gesture_recognition':
            cv2.putText(
                display_frame, 
                f'Applied filter: {current_filter}',
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2)
        elif mode == 'data_capture':
            cv2.putText(
                display_frame, 
                f'Capturing {capturing_gesture} #{capture_number}',
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2)

        cv2.imshow("MediaPipe Hands Capture", display_frame)


        # Change current filter setting if gesture is detected and we haven't changed recently
        if mode == 'gesture_recognition' and landmarks.multi_hand_landmarks and (datetime.now() - last_filter_change_time).total_seconds() > filter_change_interval:
            
            # Update the last filter change time
            last_filter_change_time = datetime.now()

            # Get the hand landmarks and begin normalization
            print("Landmarks detected")
            hand_landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in landmarks.multi_hand_landmarks[0].landmark]          

            # Get the average landmark coordinates and center the hand coordinates
            hand_landmarks_centers = []
            for dim in range(3):
                hand_landmarks_centers.append(np.mean([landmark[dim] for landmark in hand_landmarks]))
            
            for lm_id in range(len(hand_landmarks)):
                for dim in range(3):
                    hand_landmarks[lm_id][dim] -= hand_landmarks_centers[dim]
            
            # Get min and max for each dimension
            hand_landmarks_mins = []
            hand_landmarks_maxs = []
            for dim in range(3):
                hand_landmarks_mins.append(np.min([landmark[dim] for landmark in hand_landmarks]))
                hand_landmarks_maxs.append(np.max([landmark[dim] for landmark in hand_landmarks]))
                
            for lm_id in range(len(hand_landmarks)):
                for dim in range(3):
                    hand_landmarks[lm_id][dim] = (hand_landmarks[lm_id][dim] - hand_landmarks_mins[dim]) / (hand_landmarks_maxs[dim] - hand_landmarks_mins[dim])     
                    
            # Convert the 21x3 landmarks to a 1D array          
            hand_landmarks = np.array(hand_landmarks).flatten()

            # add handedness as a numeric feature, 0 for left, 1 for right
            if landmarks.multi_handedness[0].classification[0].label == 'Left':
                hand_landmarks = np.append(hand_landmarks, 0)
            else:
                hand_landmarks = np.append(hand_landmarks, 1)

            # Convert to tensor
            hand_landmarks_tensor = torch.tensor(hand_landmarks, dtype=torch.float32)
            

            # Predict gesture using the model
            with torch.no_grad():
                gesture_output = gesture_model(hand_landmarks_tensor)
                predicted_gesture = gesture_list[torch.argmax(gesture_output).item()]
            print(f"Predicted gesture: {predicted_gesture}")


            # Check if the predicted gesture is in the gesture map
            if predicted_gesture in gesture_map.keys():
                current_filter = gesture_map[predicted_gesture]
                print(f"Applying filter: {current_filter}")

        
        # Handle keyboard input 
        # masking to capture only the lower 8 bits to help with cross-platform compatibility
        key = cv2.waitKey(1) & 0xFF
        
        # 'c' key to capture current frame and landmarks
        if key == ord('c'):
            if landmarks.multi_hand_world_landmarks:
                captured_frame = display_frame.copy()

                landmarks_data = {
                    'capturing_gesture': capturing_gesture,
                    'capture_number': capture_number, 
                    'handedness': landmarks.multi_handedness[0].classification[0].label,
                    'score': landmarks.multi_handedness[0].classification[0].score
                }          
                for landmark_id in range(num_landmarks):
                    landmark = landmarks.multi_hand_landmarks[0].landmark[landmark_id]
                    landmarks_data[f'{landmark_id}_x'] = landmark.x
                    landmarks_data[f'{landmark_id}_y'] = landmark.y
                    landmarks_data[f'{landmark_id}_z'] = landmark.z
                        
                # Append landmarks data to dataframe
                landmarks_df = pd.concat([landmarks_df, pd.DataFrame([landmarks_data])], ignore_index=True)

                # Save image
                image_path = os.path.join(save_dir, f"{capturing_gesture}_{capture_number}.png")
                cv2.imwrite(image_path, captured_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                print(f"Captured landmark data and saved image to {image_path}")
                
                # Reset capture to continue with live feed
                captured_frame = None
                capture_number += 1
                print("Frame captured with hand landmarks")
            else:
                print("No hands detected. Position your hand in the frame.")
        # 's' key to save all captured landmarks to CSV then reset
        elif key == ord('s'):  
            landmarks_df.to_csv(os.path.join(save_dir, f"{capturing_gesture}_landmarks.csv"), index=False)
            print(f"Saved landmarks to {capturing_gesture}_landmarks.csv")
            landmarks_df = pd.DataFrame(columns=landmarks_header)
            capture_number = 0
        # 't' key to change capturing gesture
        elif key == ord('t'):
            capturing_gesture = gesture_list[(gesture_list.index(capturing_gesture) + 1) % len(gesture_list)]
            capture_number = 0
            print(f"Capturing gesture changed to: {capturing_gesture}")
        # 'f' key to toggle between toggling filters with gestures
        elif key == ord('f'):
            if mode == 'gesture_recognition':
                mode = 'data_capture'
                print("Switched to data capture mode")
            else:
                mode = 'gesture_recognition'
                print("Switched to gesture recognition mode")

        # 'q' key to quit
        elif key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
