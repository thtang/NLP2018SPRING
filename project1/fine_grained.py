## Python script for fine grained sentiment analysis
import json
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb

import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from afinn import Afinn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.linear_model import LinearRegression


import numpy as np 
import pandas as pd
import pickle

from sklearn.metrics import mean_squared_error




# load data
with open("./training_set.json", "r") as f:
    data = f.read()
    train_data = json.loads(data)
    
print("number of training instances:", len(train_data))

with open("./test_set.json", "r") as f:
    data = f.read()
    test_data = json.loads(data)
print("number of testing instances:", len(test_data))

#### load sentiment dictionary
# AFINN
afinn = Afinn()
# VADER
analyzer = SentimentIntensityAnalyzer()
# NTUSD-FIN
with open("./NTUSD-Fin/NTUSD_Fin_word_v1.0.json", "r") as f:
    data = f.read()
    NTUSD = json.loads(data)
word_sent_dict = {}
for i in range(len(NTUSD)):
    word_sent_dict[NTUSD[i]["token"]] = NTUSD[i]["market_sentiment"]

# load class for preprocessing
lemmatizer = WordNetLemmatizer()
tagger=PerceptronTagger()

stop_wordsstop_wo  = set(stopwords.words('english'))
stop_word= ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '@', '#']
def remove_stopwords(data):
    sentence_token = [s.split(' ') for s in data] 
    idx = 0
    for sentence in sentence_token:
        clean_sentence_token = []
        for word in sentence:
            #if word not in list(stop_words):
            word= ''.join(c for c in word if c not in stop_word)
            if word != '':
                clean_sentence_token.append(word.lower())
        sentence_token[idx] = clean_sentence_token
        idx = idx + 1
    return sentence_token

# preprocess tweets data
train_X = [item["tweet"].lower() for item in train_data]
train_X = remove_stopwords(train_X)
train_y = np.array([item["sentiment"] for item in train_data],dtype=np.float)

test_X = [item["tweet"].lower() for item in test_data]
test_X = remove_stopwords(test_X)
test_y = np.array([item["sentiment"] for item in test_data],dtype=np.float)


def get_wordnet_tag(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def get_score(sentence):
	# input: string 
	# output: sentiment score vector for that string

    sent_value = []
    sentence_tagged = np.array(nltk.pos_tag(sentence))
    for tagged in sentence_tagged:
        wn_tag = get_wordnet_tag(tagged[1])
        word = tagged[0]
        
        sentiwordnet_neg = 0.
        sentiwordnet_pos = 0.
        #get sentiwordnet score
        if wn_tag in (wn.NOUN, wn.ADJ, wn.ADV,  wn.VERB):            
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if lemma:
                synsets = wn.synsets(lemma, pos=wn_tag)
                if synsets:
                    swn_synset = swn.senti_synset(synsets[0].name())
                    sentiwordnet_neg = swn_synset.neg_score()
                    sentiwordnet_pos = swn_synset.pos_score()
    
        #get NTUSD dict score
        try: 
            dict_score = word_sent_dict[word]
        except:
            dict_score = 0.0
            
        s = " ".join(sentence)
        # affin
        afinn_score = afinn.score(s)
        
        # vader
        vader_output = analyzer.polarity_scores(s)
        vader_neg = vader_output["neg"]
        vader_pos = vader_output["pos"]
        vader_neu = vader_output["neu"]
        
        word_score = np.array([sentiwordnet_pos, sentiwordnet_neg,
                               dict_score,afinn_score, vader_neg, vader_pos], dtype=float)
        sent_value.append(word_score)
    return np.average(np.array(sent_value), axis=0)


# preparing input feature vector
train_pred = np.array([get_score(sentence) for sentence in train_X])
train_nor_pred = (((train_pred-np.min(train_pred,axis=0))/ (np.max(train_pred, axis=0)-np.min(train_pred,axis=0))) - 0.5)*2.

test_pred = np.array([get_score(sentence) for sentence in test_X])
test_nor_pred = (((test_pred-np.min(test_pred, axis=0))/ (np.max(test_pred, axis=0)-np.min(test_pred, axis=0))) - 0.5)*2.


# use xgboost as prediction model


d = 7
r = 0.1
e = 200

xbg_regr = xgb.XGBRegressor(max_depth=d,
                           learning_rate=r,
                           n_estimators=e,
                           n_jobs=-1)
xbg_regr.fit(train_pred,train_y)
xbg_regr_pred = xbg_regr.predict(test_pred)
mse = mean_squared_error(xbg_regr_pred, test_y)
               
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xbg_regr,f)
   
print("test mse:", mse)
