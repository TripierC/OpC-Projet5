#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow_hub as hub
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')


# In[ ]:


stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()
words = set(nltk.corpus.words.words())

def cleaner(sentence):

        sentence_w_punct = "".join([i.lower() for i in sentence if i not in string.punctuation])

        sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit())

        tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num)

        words_w_stopwords = [i for i in tokenize_sentence if i not in stopwords]

        words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)

        sentence_clean = ' '.join(w for w in words_lemmatize if w.lower() in words or not w.isalpha())

        return sentence_clean


# In[ ]:


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
use = hub.load(module_url)

def embed(input):
    return use(input)


# In[ ]:


model = joblib.load('model_lr')
mlb = joblib.load('multilabel_binarizer')


# Création d'une nouvelle instance fastAPI
app = FastAPI()

# Définir un objet (une classe) pour réaliser des requêtes
# dot notation (.)
class Question(BaseModel):
    body : str

# Definition du chemin du point de terminaison (API)
@app.post("/predict") # local : http://127.0.0.1:8000/predict

# Définition de la fonction de prédiction
def predict(question : Question):
    
    data = question.body
    
    data_clean = cleaner(data)
    
    features = embed([data])
    
    predict = model.predict(features)
    
    tags_predict = mlb.inverse_transform(predict)
    
    # Resultats
    
    return (tags_predict)

