# Plant Cure

## Background
Plant Cure is a plant disease detection application that aims to automate the identification and diagnosis of diseases affecting plants. This project utilizes the power of deep learning algorithms to analyze images of plants and accurately classify them as healthy or infected with a specific disease.
</p>

## About the dataset

-- Dataset credit: [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.

## Approach
A model built using pytorch was developed and tested on standard laptop CPU. Later the core training was offloaded to a GPU on google colab. 

## Performance

```
Train loss: 0.051834, Train acc:98.33267% | Test loss: 0.13384, Test acc: 96.01%
```