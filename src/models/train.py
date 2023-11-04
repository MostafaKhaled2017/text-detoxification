import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertForMaskedLM, BertTokenizer
from scipy.spatial.distance import cosine
from joblib import dump


# Reading the datadrame
df = pd.read_csv('data/interim/02_ParaNMT_train.csv')

# Step 1: Train a logistic regression classifier
# We will use the CountVectorizer to convert the text data into a bag-of-words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['reference'])
y = (df['ref_tox'] > 0.5).astype(int)  # Binary classification based on the toxicity threshold

print("Training the logistic regression model...")
# Train the logistic regression model
classifier = LogisticRegression()
classifier.fit(X, y)
print("Done training the logistic regression model.")

# Step 2: Identify toxic words
# Get the feature names and their corresponding weights from the classifier
feature_names = np.array(vectorizer.get_feature_names_out())
weights = classifier.coef_[0]

# Normalize the weights and find the indices of the words with the highest weights
normalized_weights = weights / np.linalg.norm(weights)
toxic_indices = np.argsort(normalized_weights)[-10:]  # Get top 10 toxic words for example

# Step 3: Generate potential substitutes using BERT
print("Generating substitutes using BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
print("Done generating substitutes using BERT.")

def get_substitutes(sentence, toxic_words):
    substitutes = {}
    for word in toxic_words:
        # Mask each toxic word in the sentence
        masked_sentence = sentence.replace(word, tokenizer.mask_token * len(word.split()))
        inputs = tokenizer.encode_plus(masked_sentence, return_tensors='pt')
        input_ids = inputs['input_ids']
        token_logits = model(input_ids).logits
        
        # Find all indices of the mask_token_id in input_ids
        masked_token_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        
        # If no mask_token was found, skip this word
        if len(masked_token_indices) == 0:
            # print(f"No masked token found for word '{word}' in the sentence.")
            continue

        substitutes_for_word = []
        for idx in masked_token_indices:
            # Get top 5 tokens for each masked token index
            top_5_tokens = torch.topk(token_logits[0, idx], 5).indices.tolist()
            substitutes_for_word.extend([tokenizer.decode([token]) for token in top_5_tokens])
        
        substitutes[word] = substitutes_for_word

    return substitutes

print("Getting substitutes...")
toxic_words = feature_names[toxic_indices]
substitutes = get_substitutes('example sentence with toxic words', toxic_words)
print("Done getting substitutes.")

# Step 4: Rerank the substitutes
# We will rerank based on the cosine similarity and the normalized weights as a proxy for non-toxicity
def rerank_substitutes(substitutes, sentence_embeddings, toxic_weights):
    ranked_substitutes = {}
    for word, word_substitutes in substitutes.items():
        word_index = np.where(feature_names == word)[0][0]
        word_weight = toxic_weights[word_index]
        for substitute in word_substitutes:
            sub_emb = sentence_embeddings[tokenizer.encode(substitute, add_special_tokens=False)]
            similarity = 1 - cosine(sentence_embeddings, sub_emb)
            non_toxicity_score = 1 - word_weight  # The lower the weight, the less toxic the word
            substitute_score = similarity + non_toxicity_score
            ranked_substitutes[substitute] = substitute_score
    # Sort substitutes based on the score
    return {word: sorted(subs, key=lambda x: x[1], reverse=True) for word, subs in ranked_substitutes.items()}

print("Reranking substitutes...")
ranked_substitutes = rerank_substitutes(substitutes, model.get_input_embeddings().weight.detach().numpy(), normalized_weights)
print("Done reranking substitutes.")

# # Display the ranked substitutes
# for word, subs in ranked_substitutes.items():
#     print(f"Toxic word: {word}, substitutes: {subs}")

# Save the logistic regression model to disk
dump(classifier, 'logistic_regression_toxicity_classifier.joblib')

# Save the BERT model and tokenizer to disk
model.save_pretrained('models/bert_for_masked_lm')
tokenizer.save_pretrained('models/bert_for_masked_lm')
print("Saved Models and Tokenizer to disk.")

