import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Load your trained model
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=300):  # Adjust the number of classes
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 64 * 64, 512)  # Adjust size to match architecture
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=300)  # Adjust number of classes
    model.load_state_dict(torch.load('food_image_classifier.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define image transformation (ensure it matches your training pipeline)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust to match training image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class labels (update with actual class names from your dataset)
@st.cache_resource
def load_class_names():
    return {i: f"Class_{i}" for i in range(300)}  # Replace with actual class names

class_names = load_class_names()

# Streamlit App Interface
st.title("Food Image Classification App")
st.write("Upload a food image, and the model will classify it.")

# Image Upload Section
uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Apply transformations
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)

        # Get class name
        predicted_class = class_names[predicted_idx.item()]
        st.success(f"Predicted Class: **{predicted_class}**")
    except Exception as e:
        st.error(f"Error processing image: {e}")
