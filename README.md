# Toxic Comment Classification
```
Challenge from Kaggle: The study of negative online behaviors
```
#Deep learning, NLP, text classification.

```
Solution: Build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate
```
This script runs using Python 3. 

## Dataset
Pre-trained word vectors: 840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB.

```
Pre-trained GloVe vectors (Global Vectors for Word Representation) from Stanford. 
```

## Data Pre-Processing
- Remove stopwords and punctuation: NLTK (Natural Language Toolkit).
- Make everything lowercase.
- Shuffle data and split train and validation dataset.

## Build model
- Word Embedding.
- 3-D Tensor into LSTM layer.

## Training
- Go through 2 epoches (not to overfit).
(accuracy image here)

## Evaluate model
- Calculate training and validation loss.
(image)
- Calculate training and validation accuracy.
(image)

### Final Result: 97.79% (Leaderboard: 98.85%)


*Reference: 
- https://nlp.stanford.edu/projects/glove/
- https://towardsdatascience.com/classify-toxic-online-comments-with-lstm-and-glove-e455a58da9c7
