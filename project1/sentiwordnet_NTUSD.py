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
    sent_value_dict = []
    sent_value_semiwordnet = []
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
        
        '''
        word_score = nltk_sentiwordnet_score * nltk_sentiwordnet_weight + dict_score * dict_weight    
        sent_value.append(word_score)
        '''
        sent_value_dict.append(dict_score)
        sent_value_semiwordnet.append(nltk_sentiwordnet_score)
    #print(sent_value)
    #return round(np.average(sent_value), 3)
    return round(np.average(sent_value_dict), 3), round(np.average(sent_value_semiwordnet), 3)

#################################################
#Main
#################################################
train_NTUSD_score = []
train_semiwordnet_score = []

train_sentence_token = remove_stopwords(train_X)

for sentence in train_sentence_token:
    NTUSD_score, semiwordnet_score = get_score(sentence)
    train_NTUSD_score.append(NTUSD_score)
    train_semiwordnet_score.append(semiwordnet_score)
train_NTUSD_score = np.asarray(train_NTUSD_score)    
train_semiwordnet_score = np.asarray(train_semiwordnet_score) 

train_nor_pred_NTUSD = (((train_NTUSD_score-min(train_NTUSD_score))/ (max(train_NTUSD_score)-min(train_NTUSD_score))) - 0.5)*2.   
train_nor_pred_semiwordnet = (((train_semiwordnet_score-min(train_semiwordnet_score))/ (max(train_semiwordnet_score)-min(train_semiwordnet_score))) - 0.5)*2.   

print("train mse NTUSD:", mean_squared_error(y_pred=train_nor_pred_NTUSD, y_true=train_y))
print("train mse semiwordnet:", mean_squared_error(y_pred=train_nor_pred_semiwordnet, y_true=train_y))


test_NTUSD_score = []
test_semiwordnet_score = []

test_sentence_token = remove_stopwords(test_X)

for sentence in test_sentence_token:
    NTUSD_score, semiwordnet_score = get_score(sentence)
    test_NTUSD_score.append(NTUSD_score)
    test_semiwordnet_score.append(semiwordnet_score)
test_NTUSD_score = np.asarray(test_NTUSD_score)    
test_semiwordnet_score = np.asarray(test_semiwordnet_score) 

test_nor_pred_NTUSD = (((test_NTUSD_score-min(test_NTUSD_score))/ (max(test_NTUSD_score)-min(test_NTUSD_score))) - 0.5)*2.   
test_nor_pred_semiwordnet = (((test_semiwordnet_score-min(test_semiwordnet_score))/ (max(test_semiwordnet_score)-min(test_semiwordnet_score))) - 0.5)*2.   


print("test mse NTUSD:", mean_squared_error(y_pred=test_nor_pred_NTUSD, y_true=test_y))
print("test mse semiwordnet:", mean_squared_error(y_pred=test_nor_pred_semiwordnet, y_true=test_y))

# use linear regression for mapping
lr = LinearRegression()
train_score = np.vstack((train_NTUSD_score, train_semiwordnet_score)).T
test_score = np.vstack((test_NTUSD_score, test_semiwordnet_score)).T
lr.fit(train_score, train_y)
test_lr_pred = lr.predict(test_score)
print("test mse(lr):", mean_squared_error(test_lr_pred, test_y))

'''
train mse NTUSD: 0.20275246214468043
train mse semiwordnet: 0.15387493243226374
test mse NTUSD: 0.261856032791768
test mse semiwordnet: 0.2964114412796832
test mse(lr): 0.1104597743647811
'''




'''
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
'''
'''
train mse: 0.12654722941877028
test mse: 0.13584099006011777
test mse(lr): 0.11040874465138638
'''










