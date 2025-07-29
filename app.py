import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile
import math
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- Model and Path Setup ---
MODEL_NAME = 'pose_landmarker_heavy.task'
MODEL_URL = f'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/{MODEL_NAME}'
MODEL_PATH = os.path.join(tempfile.gettempdir(), MODEL_NAME)

# --- Download the Model if it doesn't exist ---
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model to {MODEL_PATH}...")
    response = requests.get(MODEL_URL)
    response.raise_for_status()  # Ensure the download was successful
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("Download complete.")

# --- MediaPipe Pose Landmarker Setup ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_segmentation_masks=False # Not needed for our use case
)
landmarker = PoseLandmarker.create_from_options(options)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Allow requests from our React frontend
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Analysis Helper Functions ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points (e.g., elbow angle)."""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def analyze_landmarks(landmarks_per_frame):
    """Analyzes the full list of landmarks to generate feedback."""
    feedback = []
    
    # 1. Elbow Angle Analysis (Check for excessive bending)
    min_elbow_angle = 180
    for frame_landmarks in landmarks_per_frame:
        if frame_landmarks and len(frame_landmarks) > 16: # Ensure all points exist
            # For a right-handed player, check the right elbow
            right_shoulder = [frame_landmarks[12]['x'], frame_landmarks[12]['y']]
            right_elbow = [frame_landmarks[14]['x'], frame_landmarks[14]['y']]
            right_wrist = [frame_landmarks[16]['x'], frame_landmarks[16]['y']]
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            if angle < min_elbow_angle:
                min_elbow_angle = angle

    if min_elbow_angle < 90:
        feedback.append(f"Your arm is quite bent. The minimum elbow angle was {int(min_elbow_angle)}°. Try to keep the arm straighter for more power.")
    elif min_elbow_angle != 180:
        feedback.append("Good arm extension! Your elbow angle looks solid.")
    else:
        feedback.append("Could not determine elbow angle. Make sure your arm is visible.")


    # 2. Head Movement Analysis
    if landmarks_per_frame and len(landmarks_per_frame) > 1 and landmarks_per_frame[0] and landmarks_per_frame[-1]:
        # Ensure first and last frames have landmarks
        if len(landmarks_per_frame[0]) > 0 and len(landmarks_per_frame[-1]) > 0:
            initial_nose_y = landmarks_per_frame[0][0]['y']
            final_nose_y = landmarks_per_frame[-1][0]['y']
            movement = abs(initial_nose_y - final_nose_y)
            
            # Movement is a ratio of screen height, 0.05 is 5% of screen height
            if movement > 0.05:
                feedback.append("You are moving your head vertically during the stroke. Try to keep your head still to improve balance and consistency.")
            else:
                feedback.append("Excellent head stability! Keeping your head still is key.")
        else:
            feedback.append("Could not determine head movement. Ensure you are visible at the start and end of the clip.")


    return feedback


# --- API Endpoint ---
@app.route('/api/analyze', methods=['POST'])
def analyze_video_endpoint():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    
    print(f"Analyzing video: {video_path}")

    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Fallback if fps is not available from video metadata
        if fps == 0:
            fps = 30 

        all_landmarks = []
        frame_number = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the frame to MediaPipe's Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # **FIX**: Calculate timestamp manually to ensure it's monotonically increasing
            frame_timestamp_ms = int((frame_number / fps) * 1000)
            
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            frame_landmarks = []
            if pose_landmarker_result.pose_landmarks:
                for lm in pose_landmarker_result.pose_landmarks[0]:
                    frame_landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility})
            all_landmarks.append(frame_landmarks)

            frame_number += 1 # Increment frame counter

        cap.release()
        
        # Generate feedback
        feedback = analyze_landmarks(all_landmarks)
        
        return jsonify({
            'message': 'Analysis successful',
            'landmarks': all_landmarks,
            'feedback': feedback
        })

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Cleaned up {video_path}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

