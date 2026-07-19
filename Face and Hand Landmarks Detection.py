# Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['GLOG_minloglevel'] = '2'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Silence WebRTC/aioice socket closing tracebacks from stderr
import sys
class TracebackFilter:
    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, data):
        # Ignore lines from the stun retry/AttributeError loop
        if any(msg in data for msg in ["Transaction.__retry", "send_stun", "sendto", "call_exception_handler", "stun.py", "ice.py", "selector_events.py"]):
            return
        self.original_stream.write(data)

    def flush(self):
        self.original_stream.flush()

sys.stderr = TracebackFilter(sys.stderr)

import numpy as np
import cv2
import time
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Set page config
st.set_page_config(
    page_title="Face and Hand Landmarks Detection",
    page_icon="🤖",
    layout="wide"
)

# App Title & Description
st.title("🤖 Real-time Face & Hand Landmarks Detection")
st.markdown("""
This app processes your webcam feed in real-time to detect face and hand landmarks using **MediaPipe Holistic**.
""")

# Sidebar config
st.sidebar.title("Configuration")
app_mode = st.sidebar.selectbox("App Mode", ["Real-time Stream", "Snapshot Capture"])
detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.5, 0.05)
tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.5, 0.05)

show_face = st.sidebar.checkbox("Show Face Landmarks", value=True)
show_hands = st.sidebar.checkbox("Show Hand Landmarks", value=True)

# ICE Servers for WebRTC (required for deployment / hosting)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
            {"urls": ["stun:stun.services.mozilla.com"]}
        ]
    }
)

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Global settings dictionary to share configuration with the WebRTC thread
if "CONFIG" not in globals():
    CONFIG = {
        "show_face": True,
        "show_hands": True,
        "detection_confidence": 0.5,
        "tracking_confidence": 0.5,
        "model": None,
        "previous_time": time.time()
    }

# Update configurations on the main Streamlit thread execution
CONFIG["show_face"] = show_face
CONFIG["show_hands"] = show_hands
CONFIG["detection_confidence"] = detection_confidence
CONFIG["tracking_confidence"] = tracking_confidence

def get_model():
    model = CONFIG["model"]
    det = CONFIG["detection_confidence"]
    track = CONFIG["tracking_confidence"]
    
    # Check if we need to initialize or recreate the model on confidence change
    if model is None or getattr(model, "_det", None) != det or getattr(model, "_track", None) != track:
        if model is not None:
            try:
                model.close()
            except Exception:
                pass
        model = mp_holistic.Holistic(
            min_detection_confidence=det,
            min_tracking_confidence=track
        )
        # Store configuration on model object as metadata
        model._det = det
        model._track = track
        CONFIG["model"] = model
    return model

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    
    # Retrieve model (initialized lazily inside the WebRTC thread)
    model = get_model()
    
    # Converting BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process frame
    rgb_image.flags.writeable = False
    results = model.process(rgb_image)
    rgb_image.flags.writeable = True
    
    # Drawing the Facial Landmarks
    if CONFIG["show_face"] and results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(
                color=(255, 0, 255),
                thickness=1,
                circle_radius=1
            ),
            mp_drawing.DrawingSpec(
                color=(0, 255, 255),
                thickness=1,
                circle_radius=1
            )
        )

    if CONFIG["show_hands"]:
        # Drawing Right hand Land Marks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS
            )

        # Drawing Left hand Land Marks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS
            )
    
    # Calculating the FPS
    current_time = time.time()
    fps = 1 / (current_time - CONFIG["previous_time"] + 1e-6)
    CONFIG["previous_time"] = current_time
    
    # Displaying FPS on the image
    cv2.putText(image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    return av.VideoFrame.from_ndarray(image, format="bgr24")

if app_mode == "Real-time Stream":
    # Start streaming
    ctx = webrtc_streamer(
        key="landmarks",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"min": 1280, "ideal": 1920},
                "height": {"min": 720, "ideal": 1080},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True
    )
else:
    # Use st.camera_input for snapshot capture mode (works natively on Streamlit Cloud)
    img_file = st.camera_input("Take a photo to detect landmarks")
    if img_file is not None:
        # Convert file to numpy array
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # BGR
        
        # Retrieve the holistic model
        model = get_model()
        
        # Converting BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process frame
        rgb_image.flags.writeable = False
        results = model.process(rgb_image)
        rgb_image.flags.writeable = True
        
        # Drawing the Facial Landmarks
        if CONFIG["show_face"] and results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(
                    color=(255, 0, 255),
                    thickness=1,
                    circle_radius=1
                ),
                mp_drawing.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=1,
                    circle_radius=1
                )
            )

        if CONFIG["show_hands"]:
            # Drawing Right hand Land Marks
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.right_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS
                )

            # Drawing Left hand Land Marks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.left_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS
                )
        
        # Display the processed image in high quality
        st.image(image, channels="BGR", caption="Landmarks Detection Result", use_container_width=True)

# Sidebar references documentation block
st.sidebar.markdown("---")
st.sidebar.subheader("Hand Landmark References")
st.sidebar.code("\n".join([f"{l.name}: {l.value}" for l in mp_holistic.HandLandmark]))