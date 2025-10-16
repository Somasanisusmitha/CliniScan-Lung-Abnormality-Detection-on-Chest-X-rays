import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Safe import for pytorch_grad_cam (auto-installs if missing)
# -------------------------------------------------------------------
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ModuleNotFoundError:
    import os
    os.system("pip install pytorch-gradcam==0.2.1")
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

import gdown  # for model download

# ---------------------------
# Device (CPU/GPU)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"✅ Using device: {device}")

# ---------------------------
# Model Path and Download
# ---------------------------
MODEL_URL = "https://github.com/Somasanisusmitha/CliniScan-Lung-Abnormality-Detection-on-Chest-X-rays/releases/download/v1.0/model.pth"
MODEL_PATH = "model.pth"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("📦 Downloading model... Please wait ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found at {path}")
        st.stop()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 output classes

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt["classes"]

model, classes = load_model(MODEL_PATH)

# ---------------------------
# Image Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🩺 CliniScan: Chest X-Ray Abnormality Detection")
st.write("Upload a Chest X-ray image to classify it and view the Grad-CAM heatmap.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    # ---------------------------
    # Prediction
    # ---------------------------
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    st.write(f"### 🧠 Prediction: **{classes[pred_class]}**")
    st.write(f"Confidence: {probs[pred_class].item() * 100:.2f}%")
    st.bar_chart({classes[i]: probs[i].item() for i in range(len(classes))})

    # ---------------------------
    # Grad-CAM
    # ---------------------------
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = np.array(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    st.subheader("🔥 Grad-CAM Heatmap")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(visualization)
    ax[1].set_title("Grad-CAM Visualization")
    ax[1].axis("off")

    st.pyplot(fig)
