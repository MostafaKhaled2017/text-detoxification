## Name : Mostafa Kira
## Email: m.kira@innopolis.university
## Group : B21-DS02

To solve the text-detoxification problem, I did the following:
1- Train a logistic regression model to classify sentences as toxic or neutral using word features.
2- Use the learned weights to identify toxic words.
3- Generate potential substitutes for toxic words using BERT Model.
4- Rerank the substitutes based on similarity and reduced toxicity.

This approach was proposed in [the following](https://aclanthology.org/2021.emnlp-main.629.pdf) research paper