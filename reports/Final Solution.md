# Text Detoxification Task Report

## Introduction

Text detoxification is a crucial task in natural language processing aimed at mitigating the negative impacts of toxic language. The task involves identifying and neutralizing toxic content within text data, which is essential for maintaining healthy and constructive online interactions. This report outlines the approach taken to develop a text detoxification model, leveraging machine learning and natural language understanding techniques.

## Data Analysis

The ParaNMT dataset, provided with the assignment, served as the foundation for this project. A detailed investigation of the dataset's various columns was conducted to understand the importance of each feature in relation to text toxicity. The analysis focused on discerning patterns and distributions of toxic language within the dataset, which was crucial for designing the model's feature set and informing the subsequent steps of the detoxification process.

## Model Specification

The approach utilizes two models: a logistic regression model and a BERT model. The logistic regression model serves as an assisting classifier to determine whether sentences are toxic or not. Its simplicity and interpretability make it suitable for this binary classification task. Once toxicity is detected, the BERT model comes into play. This powerful language model is used for the detoxification process, generating non-toxic substitutes for the identified toxic words while maintaining the sentence's original context and meaning.

## Training Process

The logistic regression model was trained on features derived from the text to identify toxic sentences. Once trained, its predictions guided the use of the BERT model. BERT was then fine-tuned on the task of generating substitutes for the toxic words, leveraging its advanced understanding of context and language nuances.

## Evaluation

The BERT model's capability for text detoxification was evaluated using a series of steps designed to quantify the effectiveness and accuracy of the substitutions it proposed. Firstly, the model's success was measured by the mean squared error (MSE) between the original sentences' toxicity levels and those of the detoxified sentences. This metric provided a clear quantitative measure of the model's ability to reduce toxicity while maintaining the semantic content of the sentences. A low MSE indicates that the detoxified sentences closely match the original sentences in terms of toxicity levels, signifying effective detoxification.

Furthermore, the average reduction in toxicity scores after the application of BERT's substitutions offered a direct assessment of how much the model decreased toxicity across the dataset. This reduction demonstrates the practical impact of the model in cleansing toxic language from text. To ensure that the substitutions were contextually appropriate, the model's reranking process also took into account the semantic similarity between the original and proposed non-toxic words. This step was vital to preserve the original meaning and context, which is often a challenge in automated text modification tasks.

## Results

The implementation of the BERT-based text detoxification model yielded promising outcomes. The application of the model on the test dataset resulted in a reduction in the toxicity scores of the sentences, as indicated by the low mean squared error (MSE) between the original and detoxified sentences' toxicity levels.

The success of the project is further underscored by the average reduction in toxicity scores, which showcases the model's effectiveness in sanitizing language. This reduction was consistent across a variety of sentence structures and topics, illustrating the model's robustness and adaptability. Overall, the project achieved its objective of creating a safer and more respectful communication environment by effectively reducing toxic language in text data.


