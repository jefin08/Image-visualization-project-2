# Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['GLOG_minloglevel'] = '2'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

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

class HolisticProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic_model = mp_holistic.Holistic(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.previous_time = time.time()
        self.show_face = show_face
        self.show_hands = show_hands

    def update_params(self, show_face, show_hands):
        self.show_face = show_face
        self.show_hands = show_hands

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        
        # Converting from BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process frame
        rgb_image.flags.writeable = False
        results = self.holistic_model.process(rgb_image)
        rgb_image.flags.writeable = True
        
        # Drawing the Facial Landmarks
        if self.show_face and results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(
                    color=(255,0,255),
                    thickness=1,
                    circle_radius=1
                ),
                mp_drawing.DrawingSpec(
                    color=(0,255,255),
                    thickness=1,
                    circle_radius=1
                )
            )

        if self.show_hands:
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
        fps = 1 / (current_time - self.previous_time + 1e-6)
        self.previous_time = current_time
        
        # Displaying FPS on the image
        cv2.putText(image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Start streaming
ctx = webrtc_streamer(
    key="landmarks",
    video_processor_factory=HolisticProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    },
    async_processing=True
)

if ctx.video_processor:
    ctx.video_processor.update_params(show_face=show_face, show_hands=show_hands)

# Keep code to access landmarks as a reference/documentation block at the end if needed,
# or simply print instructions.
st.sidebar.markdown("---")
st.sidebar.subheader("Hand Landmark References")
st.sidebar.code("\n".join([f"{l.name}: {l.value}" for l in mp_holistic.HandLandmark]))