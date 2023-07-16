import json
import torch
import pathlib
from PIL import Image
import streamlit as st
from PIL import Image
from torchvision import transforms
from model import PlantDiseaseDetector

categories = {}
with open("categories.json", "r") as f:
    categories = json.load(f)
reversed_categories = {v:k for k,v in categories.items()}

MODEL_PATH = pathlib.Path("models/plantcure_v003.pt")
NUMBER_OF_CLASSES = len(categories)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = PlantDiseaseDetector(NUMBER_OF_CLASSES).to(torch.device("cpu"))
model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))

img_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.title("PlantCure v0.1.0")

file = st.file_uploader(label="Upload Image")

if file is not None:
    image = img_transforms(Image.open(file)).to(torch.device('cpu')).unsqueeze(dim=0)
    model.eval()
    with torch.inference_mode():
        logits = model(image)
        pred_index = torch.softmax(logits,dim=1).argmax(dim=1)
        predicted_category = reversed_categories[pred_index.item()].replace("___"," ").replace("_"," ").replace(",","")
        st.markdown(f"Predicted Category: **{predicted_category}**")
    st.image(file)
    