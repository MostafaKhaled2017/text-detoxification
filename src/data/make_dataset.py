import requests
import zipfile
import os
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_link = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
file_path = 'data/raw/filtered_paranmt.zip'


print("Data Loading and Processing in Progress ...")

# Download dataset and save it to the data folder
response = requests.get(dataset_link, stream=True)
with open(file_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)
        
extract_dir = 'data/raw/filtered_paranmt/'

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    
os.remove(file_path)
    
df = pd.read_csv('data/raw/filtered_paranmt/filtered.tsv', sep='\t')

# Drop the "Unnamed: 0" column
df.drop(['Unnamed: 0'], axis=1, inplace= True)

# Drop the duplicates
df.drop_duplicates(inplace=True)

# Saving the updated dataset
df.to_csv('data/interim/01_ParaNMT_cleaned.csv', index=False)

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the data
train_df.to_csv('data/interim/02_ParaNMT_train.csv', index=False)
val_df.to_csv('data/interim/02_ParaNMT_val.csv', index=False)
