import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import psycopg2
import datetime
import os

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=False)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Adjusted for MNIST input
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Modify final layer for 10 classes
model.load_state_dict(torch.load("resnet18_mnist.pth", map_location=device))
model.to(device)
model.eval()

# Database connection setup
DB_CONN = {
    "dbname": os.getenv("DB_NAME", "mnist_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def log_prediction(predicted_digit, true_label):
    """Log predictions into PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONN)
        cur = conn.cursor()
        cur.execute("INSERT INTO predictions (timestamp, predicted_digit, true_label) VALUES (%s, %s, %s)",
                    (datetime.datetime.now(), predicted_digit, true_label))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")

def get_logging_history():
    """Retrieve and display the logging history from the PostgreSQL database with selected columns."""
    try:
        conn = psycopg2.connect(**DB_CONN)
        cur = conn.cursor()
        cur.execute("SELECT timestamp, predicted_digit, true_label FROM predictions ORDER BY timestamp DESC")
        records = cur.fetchall()
        cur.close()
        conn.close()
        return records
    except Exception as e:
        st.error(f"Database error: {e}")
        return []

def predict(image):
    #"""Preprocess the image and make a prediction."""
    image = Image.fromarray(image)
    #create the transform process (NO augmentation) used before training the classifier
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel image
        transforms.Resize((28, 28)),  # Ensure size matches MNIST format
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize like during training
    ])   
    #Preprocess the image using the transform process
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_prob = probabilities[0, predicted_class].item()  # Get the probability of the predicted digit

    return predicted_class, predicted_prob

# Streamlit UI
st.title("Digit Recognizer")

st.markdown(
    """
    <style>
        /* Hide dropdown arrows in Streamlit */
        [data-testid="stArrow"], 
        [data-baseweb="select"] > div:nth-child(2) {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Drawable canvas for digit input with black background
canvas = st_canvas(
    fill_color=None, #not needed for freedraw mode
    stroke_width=20,
    stroke_color="white",  # Ensure drawing is visible on black background
    background_color="black",  # Set background to black
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)


# User input for true label
true_label = st.number_input("Enter the true label (0-9):", min_value=0, max_value=9, step=1)

if st.button("Predict"):

    #convert canvasresult to gray scale image object and pass the image object to predict function
    if canvas.image_data is not None:
        # Convert canvas to grayscale image
        image = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    
        # Predict the digit
        predicted_digit, predicted_prob = predict(image)
        st.write(f"Predicted Digit: {predicted_digit}")
        st.write(f"Confidence: **{predicted_prob:.4f}**")
        
        # Log into PostgreSQL
        log_prediction(predicted_digit, true_label)
        st.success("Prediction logged successfully!")

if st.button("Show Logging History"):
    logs = get_logging_history()
    if logs:
        import pandas as pd
        df = pd.DataFrame(logs, columns=["Timestamp", "Predicted Digit", "True Label"])
        st.dataframe(df)
    else:
        st.write("No logs available.")
