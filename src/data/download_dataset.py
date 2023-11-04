import requests
import zipfile
import os

dataset_link = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
file_path = 'data/raw/filtered_paranmt.zip'

# Download dataset and save it to the data folder
response = requests.get(dataset_link, stream=True)
with open(file_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)
        
extract_dir = 'data/raw/filtered_paranmt/'

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    
os.remove(file_path)
    
print("Data loaded successfully")