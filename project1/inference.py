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

import pickle
import sys
from sklearn.metrics import mean_squared_error
lemmatizer = WordNetLemmatizer()
tagger=PerceptronTagger()


# input 
input_sentence = sys.argv[1]


# load sentiment dictionary 
with open("./SCL-NMA.txt", "r") as f:
    MaxDiff = f.readlines()
    MaxDiff_dict = {}
    for line in MaxDiff:
        MaxDiff_dict[line.strip().split("\t")[0]] = line.strip().split("\t")[1]

afinn = Afinn()
analyzer = SentimentIntensityAnalyzer()

# load NTUSD
with open("./NTUSD-Fin/NTUSD_Fin_word_v1.0.json", "r") as f:
    data = f.read()
    NTUSD = json.loads(data)

word_sent_dict = {}
for i in range(len(NTUSD)):
    word_sent_dict[NTUSD[i]["token"]] = NTUSD[i]["market_sentiment"]
    
stop_words = set(stopwords.words('english'))
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

def sentByDict(sent):
    sent = sent.split()
    sent_value = []
    for s in sent:
        try: 
            sent_value.append(word_sent_dict[s])
        except:
            pass
    if sent_value==[]:
        return 0
    else:
        return np.average(sent_value)
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
    sent_value = []
    #print(sentence)
    sentence_tagged = np.array(nltk.pos_tag(sentence))
    #print(sentence_tagged)
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
        
        word_score = np.array([sentiwordnet_pos,sentiwordnet_neg , dict_score,afinn_score, vader_neg, vader_pos], dtype=float)
        sent_value.append(word_score)
    #print(sent_value)
    return np.average(np.array(sent_value), axis=0)

sentence = remove_stopwords([input_sentence])

test_pred = get_score(sentence[0])

# load model
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
    
prediction = xgb_model.predict([test_pred])
print("fine-grained sentiment score:",prediction[0])