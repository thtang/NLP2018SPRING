# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LinearRegression
import numpy as np 
import pickle
from sklearn.metrics import mean_squared_error
lemmatizer = WordNetLemmatizer()
tagger=PerceptronTagger()
#nltk.download('stopwords')
#nltk.download('sentiwordnet')
#################################################
#read data
#################################################

stop_words = set(stopwords.words('english'))
stop_word= ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '@', '#']

train_X = np.load('data/train_X.npy')
train_y = np.load('data/train_y.npy')
test_X = np.load('data/test_X.npy')
test_y = np.load('data/test_y.npy')
with open('dict/word_sent_dict.pkl', 'rb') as f:
    word_sent_dict = pickle.load(f)
    
    
#################################################
#data preprocessing
#################################################
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
    

#################################################
#Calculation
#################################################
dict_weight = 0.3
nltk_sentiwordnet_weight = 0.7

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
        
        nltk_sentiwordnet_score = 0
        #get sentiwordnet score
        if wn_tag in (wn.NOUN, wn.ADJ, wn.ADV,  wn.VERB):            
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if lemma:
                synsets = wn.synsets(lemma, pos=wn_tag)
                if synsets:
                    swn_synset = swn.senti_synset(synsets[0].name())
                    nltk_sentiwordnet_score = swn_synset.pos_score() - swn_synset.neg_score()
    
        #get NTUSD dict score
        try: 
            dict_score = word_sent_dict[word]
        except:
            dict_score = 0
        
        word_score = nltk_sentiwordnet_score * nltk_sentiwordnet_weight + dict_score * dict_weight    
        sent_value.append(word_score)
    #print(sent_value)
    return round(np.average(sent_value), 3)


#################################################
#Main
#################################################
train_sentence_token = remove_stopwords(train_X)
train_pred = np.array([get_score(sentence) for sentence in train_sentence_token],dtype=np.float)
train_nor_pred = (((train_pred-min(train_pred))/ (max(train_pred)-min(train_pred))) - 0.5)*2.        
print("train mse:", mean_squared_error(y_pred=train_nor_pred, y_true=train_y))


test_sentence_token = remove_stopwords(test_X)
test_pred = np.array([get_score(sentence) for sentence in test_sentence_token],dtype=np.float)
test_nor_pred = (((test_pred-min(test_pred))/ (max(test_pred)-min(test_pred))) - 0.5)*2.        
print("test mse:", mean_squared_error(y_pred=test_nor_pred, y_true=test_y))


# use linear regression for mapping
lr = LinearRegression()
lr.fit(train_pred.reshape(-1,1), train_y)
test_lr_pred = lr.predict(test_pred.reshape(-1,1))
print("test mse(lr):", mean_squared_error(test_lr_pred, test_y))












