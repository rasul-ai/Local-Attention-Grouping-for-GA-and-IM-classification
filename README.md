# Local-Attention-Grouping-for-GA-and-IM-classification
**Author:** [Md Rasul Islam Bapary]  
**Date:** [10.01.2024]

In this repository I have tried to implement LAG model of my own.I have inspired to implement this from a paper ***A Benchmark Dataset of Endoscopic Images and Novel Deep Learning Method to Detect Intestinal Metaplasia and Gastritis Atrophy*** [Link](https://pubmed.ncbi.nlm.nih.gov/36306301/). I have trained and tested the model ***[Link](https://github.com/rasul-ai/Local-Attention-Grouping-for-GA-and-IM-classification/blob/main/LAG_model.ipynb)*** with Cats and Dogs dataset. I have contacted with original author for the dataset that they had used. But they did not response. The original model architecture is this.

![Model_Architecture](https://github.com/rasul-ai/Local-Attention-Grouping-for-GA-and-IM-classification/blob/main/Images/GA_IM.jpg)


My model have provided a good result on classifying Cats and Dogs. I will explore it further if I got the dataset from the author of the Original Paper.
### I have provided the dataset on dataset folder in this repository.

# Requirements
```
pytorch==1.7+cuda10.1
torchvision==0.6.0
numpy==1.19.5

### Please run the code on colab if you do not have local PC with GPU.
```
