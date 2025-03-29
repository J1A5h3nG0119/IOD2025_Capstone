import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os
import matplotlib.pyplot as plt

# === 1. Load Trained Model ===
@st.cache_resource
def load_cnn_lstm_model():
    return load_model("model/best_model_subject_wise.keras")

model = load_cnn_lstm_model()

# === 2. Load Haar Cascade ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === 3. Extract All Frames from Video ===
def extract_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# === 4. Detect and Crop Face ===
def detect_and_crop_face(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None

    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    face_crop = gray_frame[y:y+h, x:x+w]
    return face_crop

# === 5. Preprocess Face ===
def preprocess_face(face_crop, size=(96, 96)):
    resized = cv2.resize(face_crop, size)
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=-1)

# === 6. Prepare Sequence of Faces ===
def prepare_sequence(frames, max_frames=30):
    processed = []
    cropped_faces = []

    for frame in frames:
        face = detect_and_crop_face(frame)
        if face is not None:
            processed_face = preprocess_face(face)
            processed.append(processed_face)
            cropped_faces.append(face)

    if len(processed) < max_frames:
        return None, []

    sequence = np.array(processed[:max_frames])
    sequence = sequence[np.newaxis, ...]
    return sequence, cropped_faces[:max_frames]

# === 7. Predict from One Sequence ===
def predict_drowsiness(sequence):
    if sequence is None or sequence.shape[1] != 30:
        return "⚠️ Invalid sequence", 0.0

    try:
        pred = model.predict(sequence)
        score = float(pred[0][0])
        label = "😴 Drowsy" if score > 0.5 else "😊 Alert"
        return label, score
    except Exception as e:
        print(f"Prediction error: {e}")
        return "⚠️ Prediction failed", 0.0

# === 8. Sliding Window + Averaging ===
def sliding_window_predictions(frames, window_size=30, step=10):
    scores = []
    valid_sequences = 0
    all_cropped_faces = []

    for start in range(0, len(frames) - window_size + 1, step):
        window = frames[start:start + window_size]
        sequence, cropped_faces = prepare_sequence(window)

        if sequence is not None:
            _, score = predict_drowsiness(sequence)
            scores.append(score)
            valid_sequences += 1
            all_cropped_faces.extend(cropped_faces)

    if valid_sequences == 0:
        return None, 0.0, []

    average_score = sum(scores) / len(scores)
    final_label = "😴 Drowsy" if average_score > 0.5 else "😊 Alert"
    return final_label, average_score, all_cropped_faces

# === 9. Probability Bar ===
def draw_probability_bar(probability):
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.barh([""], [probability*100], color="skyblue")
    ax.set_xlim([0, 1])
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Drowsiness Probability (%)")
    st.pyplot(fig)

# === 10. Streamlit UI ===
st.title("🚗 Driver Drowsiness Detection")
st.markdown("Upload a short video (30 FPS).")

uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.video(uploaded_file)
    st.info("Processing video... extracting frames, running predictions...")

    frames = extract_all_frames(tmp_path)
    label, confidence, cropped_faces = sliding_window_predictions(frames)

    if not cropped_faces:
        st.error("🚫 No valid face sequences detected. Try again with a clearer front-facing video.")
    else:
        st.success(f"Final Prediction: **{label}** (Average Confidence: {confidence*100:.0f}%)")

        st.markdown("### 🔍 Cropped Face from First 30 Frames")
        cols = st.columns(5)
        for i, face_img in enumerate(cropped_faces[:30]):  # Show first 30 samples
            with cols[i % 5]:
                st.image(face_img, caption=f"Frame {i+1}", channels="GRAY", use_container_width=True)

        st.markdown("### 📊 Average Drowsiness Probability")
        draw_probability_bar(confidence)

    os.remove(tmp_path)
