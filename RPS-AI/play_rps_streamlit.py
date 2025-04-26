import streamlit as st
import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model

# Load model
model = load_model('rps_model.h5')

# Labels
labels = ['rock', 'paper', 'scissors']

# Game logic
def get_winner(user_move, ai_move):
    if user_move == ai_move:
        return "Tie!"
    elif (user_move == 'rock' and ai_move == 'scissors') or \
         (user_move == 'paper' and ai_move == 'rock') or \
         (user_move == 'scissors' and ai_move == 'paper'):
        return "You Win!"
    else:
        return "AI Wins!"

# Streamlit UI
st.title("Rock-Paper-Scissors with Computer Vision 🎮🧠")
st.text("Press the 'Play' button to capture your move!")

# Start webcam
camera = st.camera_input("Take a picture")  # this automatically opens webcam

if camera is not None:
    # When a picture is taken
    file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame = cv2.flip(frame, 1)

    # Preprocess
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img / 255.0, axis=0)

    # Predict user's move
    prediction = model.predict(img, verbose=0)
    user_move = labels[np.argmax(prediction)]

    # AI move
    ai_move = random.choice(labels)

    # Determine result
    result = get_winner(user_move, ai_move)

    # Display everything
    st.image(frame, caption='Your Move', channels="BGR")
    st.write(f"**Your Move:** {user_move}")
    st.write(f"**AI Move:** {ai_move}")
    st.success(f"**Result:** {result}")
