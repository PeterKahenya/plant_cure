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

st.markdown("""
Plant Cure is a plant disease detection application that aims to automate the identification and diagnosis of diseases affecting plants.
""")

st.sidebar.markdown("""
# Plant Cure v0.1.0


## About the dataset

-- Dataset credit: [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.

## Approach
A model built using pytorch was developed and tested on standard laptop CPU. Later the core training was offloaded to a GPU on google colab. 

## Performance

```
Train loss: 0.051834, Train acc:98.33267% | Test loss: 0.13384, Test acc: 96.01%
```
""")

file = st.file_uploader(label="Upload Plant Leaf Image")

if file is not None:
    image = img_transforms(Image.open(file)).to(torch.device('cpu')).unsqueeze(dim=0)
    model.eval()
    with torch.inference_mode():
        logits = model(image)
        pred_index = torch.softmax(logits,dim=1).argmax(dim=1)
        predicted_category = reversed_categories[pred_index.item()].replace("___"," ").replace("_"," ").replace(",","")
        plant = predicted_category.split()[0]
        disease = " ".join(predicted_category.split()[1:])
        st.markdown(f"### PLANT: **{plant}**")
        st.markdown(f"### DISEASE: **{disease}**")

    st.image(file)
    