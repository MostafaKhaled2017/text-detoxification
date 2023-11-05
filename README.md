# Text-Detoxification

This repository is dedicated to the task of text-detoxification, aiming to identify and neutralize toxic language in text. It is based on a logistic regression model that classifies sentences as toxic or neutral, using word features and BERT Model to suggest non-toxic substitutes.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before running the project, ensure you have the following installed:
- Python 3.6 or above
- pip

### Installing

Clone the repository to your local machine:

```
git clone https://github.com/MostafaKhaled2017/text-detoxification.git
cd text-detoxification
```

### Install the required packages
```
pip install -r requirements.txt
```

## Usage
The repository consists of several components that can be run sequentially to perform data transformation, model training, and predictions.

**To get the Data:**
```
python src/data/make_dataset.py
```

**To train the models:**
```
python src/models/train.py
```

**To check model predictions:**
```
python src/models/predict.py
```

**To build Visualizations:**
```
python src/models/visualize.py
```

## Research
This approach is based on the methodology proposed in the paper [Text Detoxification using Large Pre-trained Neural Models](https://aclanthology.org/2021.emnlp-main.629.pdf)

## Author
* **Name:** Mostafa Kira
* **Email:** m.kira@innopolis.university
* Studies in group B21-DS02 at Innopolis University