import torch
from joblib import load
from transformers import BertForMaskedLM, BertTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import cosine

# Load the saved logistic regression classifier and vectorizer
classifier = load('models/logistic_regression_toxicity_classifier.joblib')

# Load the BERT model and tokenizer
model = BertForMaskedLM.from_pretrained('models/bert_for_masked_lm')
tokenizer = BertTokenizer.from_pretrained('models/bert_for_masked_lm')

# Load your training data
train_df = pd.read_csv('data/interim/02_ParaNMT_train.csv')

# Fit the vectorizer on the training data to recreate the vocabulary
vectorizer = CountVectorizer()
vectorizer.fit(train_df['reference'])

# Read the external test dataset
test_df = pd.read_csv('data/interim/02_ParaNMT_val.csv')

# Sample 50 rows from the test dataset for evaluation
test_df = test_df.sample(n=50, random_state=1)

# Get the feature names from the vectorizer
feature_names = np.array(vectorizer.get_feature_names_out())

# Function to predict the toxicity of a sentence and suggest non-toxic substitutes
def predict_toxicity(sentence):
    # Vectorize the sentence
    sentence_vectorized = vectorizer.transform([sentence])

    # Predict the toxicity
    is_toxic = classifier.predict(sentence_vectorized)[0]

    # If sentence is toxic, identify toxic words and suggest substitutes
    if is_toxic:
        # Get the feature names and their corresponding weights from the classifier
        feature_names = np.array(vectorizer.get_feature_names_out())
        weights = classifier.coef_[0]

        # Normalize the weights and find the indices of the words with the highest weights
        normalized_weights = weights / np.linalg.norm(weights)
        toxic_indices = np.argsort(normalized_weights)[-10:]  # Get top 10 toxic words for example
        toxic_words = feature_names[toxic_indices]

        # Generate substitutes for toxic words
        substitutes = get_substitutes(sentence, toxic_words)
                
        # When calling rerank_substitutes, make sure to pass the toxic_weights
        toxic_weights = calculate_toxic_weights(classifier, feature_names)
        ranked_substitutes = rerank_substitutes(substitutes, sentence, toxic_weights, model, tokenizer)

        return is_toxic, ranked_substitutes
    else:
        return is_toxic, {}

def get_substitutes(sentence, toxic_words):
    substitutes = {}
    for word in toxic_words:
        # We need to escape special regex characters in words to avoid unintended behavior
        word_regex = re.escape(word)
        
        # Replace all occurrences of the toxic word with the mask token
        masked_sentence = re.sub(f"\\b{word_regex}\\b", tokenizer.mask_token, sentence)
        
        # Encode the masked sentence
        inputs = tokenizer(masked_sentence, return_tensors='pt')
        input_ids = inputs['input_ids']
        
        # Run the model to get token logits
        token_logits = model(input_ids).logits
        
        # Find the indices of the masked tokens
        masked_token_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        # Check if masked_token_indices is not empty
        if not masked_token_indices.numel():
            # print(f"No masked token found for the word '{word}' in the sentence.")
            continue
        
        # For each masked index, get the top 5 candidate substitutions
        substitutes[word] = []
        for idx in masked_token_indices:
            top_5_tokens = torch.topk(token_logits[0, idx], 5).indices.tolist()
            substitutes[word].extend([tokenizer.decode([token]) for token in top_5_tokens])

    return substitutes

# This function should be called within a scope where 'classifier' is available
def calculate_toxic_weights(classifier, feature_names):
    # Extract the coefficients from the classifier and normalize them
    weights = classifier.coef_[0]
    normalized_weights = weights / np.linalg.norm(weights)
    
    # Create a dictionary mapping feature names to their weights
    toxic_weights = {feature_names[i]: normalized_weights[i] for i in range(len(feature_names))}
    
    return toxic_weights


def rerank_substitutes(substitutes, sentence, toxic_weights, model, tokenizer):
    ranked_substitutes = {}
    sentence_embedding = model.get_input_embeddings()(tokenizer.encode(sentence, return_tensors='pt')).detach().mean(dim=1)
    
    for word, word_substitutes in substitutes.items():
        word_index = np.where(feature_names == word)[0][0]
        word_weight = toxic_weights.get(word, 0)  # Default to 0 if the word is not found

        
        for substitute in word_substitutes:
            # Encode the substitute word and get the embeddings
            sub_ids = tokenizer.encode(substitute, add_special_tokens=False)
            sub_ids_tensor = torch.tensor(sub_ids).unsqueeze(0)  # Convert to tensor and add batch dimension
            sub_emb = model.get_input_embeddings()(sub_ids_tensor).detach().mean(dim=1)

            # Compute cosine similarity
            similarity = 1 - cosine(sentence_embedding.squeeze().numpy(), sub_emb.squeeze().numpy())
            
            non_toxicity_score = 1 - word_weight  # The lower the weight, the less toxic the word
            substitute_score = similarity + non_toxicity_score
            if substitute not in ranked_substitutes:
                ranked_substitutes[substitute] = substitute_score
            else:
                ranked_substitutes[substitute] = max(ranked_substitutes[substitute], substitute_score)

    # Sort substitutes based on the score
    return {word: sorted([(sub, score) for sub, score in ranked_substitutes.items()], key=lambda x: x[1], reverse=True) for word in substitutes}


def apply_substitutes(sentence, substitutes):
    for word, sub_list in substitutes.items():
        if sub_list:  # If there are substitutes available for the word
            # Take the top substitute
            top_substitute = sub_list[0][0]  
            sentence = sentence.replace(word, top_substitute)
    return sentence

def evaluate_detoxification(test_df, model, tokenizer, vectorizer):
    original_toxicities = []
    detoxified_toxicities = []

    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        original_sentence = row['reference']
        original_toxicity = row['ref_tox']
        original_toxicities.append(original_toxicity)

        # Predict toxicity and get substitutes
        _, substitutes = predict_toxicity(original_sentence)

        # Apply the top substitutes to the original sentence
        detoxified_sentence = apply_substitutes(original_sentence, substitutes)

        # Vectorize the detoxified sentence and predict its toxicity
        detoxified_vectorized = vectorizer.transform([detoxified_sentence])
        detoxified_toxicity = classifier.predict_proba(detoxified_vectorized)[0][1]
        detoxified_toxicities.append(detoxified_toxicity)

    # Calculate the change in toxicity as a measure of detoxification success
    change_in_toxicity = np.array(original_toxicities) - np.array(detoxified_toxicities)
    mse = mean_squared_error(original_toxicities, detoxified_toxicities)
    avg_reduction = np.mean(change_in_toxicity)

    return mse, avg_reduction, change_in_toxicity

# Example usage:
if __name__ == "__main__":
    mse, avg_reduction, change_in_toxicity = evaluate_detoxification(test_df, model, tokenizer, vectorizer)
    print(f"Mean Squared Error between original and detoxified toxicities: {mse}")
    print(f"Average reduction in toxicity: {avg_reduction}")
    print(f"Toxicity changes: {change_in_toxicity}")