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

This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. 

### Supported Plant Diseases
```
{'apple black rot',
 'apple cedar apple rust',
 'apple healthy',
 'apple scab',
 'blueberry healthy',
 'cherry (including sour) healthy',
 'cherry (including sour) powdery mildew',
 'corn (maize) cercospora leaf spot gray leaf spot',
 'corn (maize) common rust',
 'corn (maize) healthy',
 'corn (maize) northern leaf blight',
 'grape black rot',
 'grape esca (black measles)',
 'grape healthy',
 'grape leaf blight (isariopsis leaf spot)',
 'orange haunglongbing (citrus greening)',
 'peach bacterial spot',
 'peach healthy',
 'pepper bell bacterial spot',
 'pepper bell healthy',
 'potato early blight',
 'potato healthy',
 'potato late blight',
 'raspberry healthy',
 'soybean healthy',
 'squash powdery mildew',
 'strawberry healthy',
 'strawberry leaf scorch',
 'tomato bacterial spot',
 'tomato early blight',
 'tomato healthy',
 'tomato late blight',
 'tomato leaf mold',
 'tomato mosaic virus',
 'tomato septoria leaf spot',
 'tomato spider mites two-spotted spider mite',
 'tomato target spot',
 'tomato yellow leaf curl virus'}          
```
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
    