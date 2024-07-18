import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import av
from PIL import Image, ImageEnhance, ImageFilter


st.title("プリクラアプリ")
st.write("2024海城祭")

# MediaPipeの設定
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# 目のランドマークのインデックス（左目と右目）
LEFT_EYE_INDEXES = [362, 258, 259, 446, 254, 252]
RIGHT_EYE_INDEXES = [226, 29, 28, 133, 22, 24]

def enlarge_eye(image, landmarks, eye_indexes, scale=1.2):
    # 目のランドマークを取得
    eye_landmarks = np.array([landmarks[idx] for idx in eye_indexes])
    
    # 目の外接矩形を計算
    min_x = int(np.min(eye_landmarks[:, 0]))
    max_x = int(np.max(eye_landmarks[:, 0]))
    min_y = int(np.min(eye_landmarks[:, 1]))
    max_y = int(np.max(eye_landmarks[:, 1]))
    
    # 目の領域を抽出
    eye_region = image[min_y:max_y, min_x:max_x]
    
    # 目の領域を拡大
    enlarged_eye = cv2.resize(eye_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # 拡大した目の領域を元の画像に貼り付け
    new_min_x = min_x - (enlarged_eye.shape[1] - (max_x - min_x)) // 2
    new_max_x = new_min_x + enlarged_eye.shape[1]
    new_min_y = min_y - (enlarged_eye.shape[0] - (max_y - min_y)) // 2
    new_max_y = new_min_y + enlarged_eye.shape[0]
    
    new_min_x = max(new_min_x, 0)
    new_min_y = max(new_min_y, 0)
    new_max_x = min(new_max_x, image.shape[1])
    new_max_y = min(new_max_y, image.shape[0])
    
    image[new_min_y:new_max_y, new_min_x:new_max_x] = enlarged_eye[:new_max_y-new_min_y, :new_max_x-new_min_x]
    return image


def capture_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in face_landmarks.landmark]
                    
            frame = enlarge_eye(frame, landmarks, LEFT_EYE_INDEXES)
            frame = enlarge_eye(frame, landmarks, RIGHT_EYE_INDEXES)
            return frame
    else:
        return frame

    

def apply_filter(image_array):
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.5)  # Adjust brightness
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(0.7)
    image = image.filter(ImageFilter.SMOOTH_MORE)  # Apply blur filter
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def callback(frame):
    image = frame.to_ndarray(format="bgr24")
    image = capture_image(image)
    filtered_image = apply_filter(image)
    return av.VideoFrame.from_ndarray(filtered_image, format="bgr24")

webrtc_streamer(
    key="kaijo_puri",
    video_frame_callback=callback,
    rtc_configuration={  # この設定を足す
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
