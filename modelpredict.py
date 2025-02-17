#load the model to take as input an image of hand written character and output the prediction of its label, the model's probability of its prediction and the true label of the input image. 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
def load_model(path="resnet18_mnist.pth"):
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Adjust for MNIST
    model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model

# Preprocessing function for input images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel image
        transforms.Resize((28, 28)),  # Ensure size matches MNIST format
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize like during training
    ])
    
    image = Image.open(image_path).convert("L")  # Open image as grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to extract the true label from the filename (optional)
def get_true_label_from_filename(image_path):
    filename = os.path.basename(image_path)
    label = filename.split("_")[0]  # Assumes format "digit_xxx.png"
    return int(label) if label.isdigit() else None

# Inference function
def predict(model, image_path):
    image = preprocess_image(image_path)
    true_label = get_true_label_from_filename(image_path)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    print(f"True Label: {true_label if true_label is not None else 'Unknown'}, "
          f"Predicted Label: {predicted_class}, Confidence: {confidence:.4f}")
    
    return true_label, predicted_class, confidence

# Load the model
model = load_model("./model/resnet18_mnist.pth")

# Example usage: Provide an image path of a handwritten digit (e.g., "7_test.png" where 7 is the true label)
image_path = "./0_test.png"
true_label, predicted_label, confidence = predict(model, image_path)
image_path = "./1_test.png"
true_label, predicted_label, confidence = predict(model, image_path)
image_path = "./2_test.png"
true_label, predicted_label, confidence = predict(model, image_path)
image_path = "./3_test.png"
true_label, predicted_label, confidence = predict(model, image_path)
image_path = "./4_test.png"
true_label, predicted_label, confidence = predict(model, image_path)
image_path = "./5_test.png"
true_label, predicted_label, confidence = predict(model, image_path)
image_path = "./6_test.png"
true_label, predicted_label, confidence = predict(model, image_path)
image_path = "./7_test.png"
true_label, predicted_label, confidence = predict(model, image_path)
image_path = "./8_test.png"
true_label, predicted_label, confidence = predict(model, image_path)
image_path = "./9_test.png"
true_label, predicted_label, confidence = predict(model, image_path)
