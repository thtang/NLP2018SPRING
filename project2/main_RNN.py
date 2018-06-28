import numpy as np
import pandas as pd
import re
import gensim
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

TRAINING_FILE_PATH = "./data/TRAIN_FILE.txt"
TESTING_FILE_PATH = "./data/TEST_FILE.txt"
preps = [' of ',' in ',' into ',' onto ',' on ',' from ',' with ',' by ',' to ',
         ' has ',' have ',' is ',' are ']

classname = []
with open("./data/classname.txt", "r", encoding='utf-8-sig') as f:
    classname = f.read().split('\n')
        
model = MultinomialNB(alpha=0.04)
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, ' ', raw_html)
  return cleantext

def ngramtolist(ngram):
    words = []
    for c in ngram:
        word = ""
        for w in c:
            word += w
        words.append(word)
    return words

def preprocessing(filepath):
    #stop_words = set(stopwords.words('english'))
    s_rex = re.compile(r'<e1>(.*?)</e2>|<e2>(.*?)</e1>',re.S|re.M)
    e1_rex = re.compile(r'<e1>(.*?)</e1>',re.S|re.M)
    e2_rex = re.compile(r'<e2>(.*?)</e2>',re.S|re.M)
    stemmer = gensim.parsing.porter.PorterStemmer()
    lmtzr = WordNetLemmatizer()
    with open(filepath, "r") as f:
        rows = f.read().split('\n')
        count = 0
        if filepath==TRAINING_FILE_PATH:
            features,e1s,e2s,dists,relations = [],[],[],[],[]
            for r in rows:
                if r and count%4 == 0:
                    feature = []
                    sentence = r.split('\t')[1]
                    sentence = s_rex.search(sentence).group(0)
                    e1 = e1_rex.search(sentence).group(0)
                    e2 = e2_rex.search(sentence).group(0)
                    sentence = cleanhtml(sentence)
                    for s in preps:
                        sentence = sentence.replace(s,s+' '+s)
                    e1 = cleanhtml(e1)
                    e2 = cleanhtml(e2)
                    word_tokens = word_tokenize(sentence)
                    sentence = [w.lower() for w in word_tokens if w.isalpha()]
                    sentence = [lmtzr.lemmatize(i,'v') for i in sentence]
                    e1 = [lmtzr.lemmatize(i,'v') for i in e1]
                    e2 = [lmtzr.lemmatize(i,'v') for i in e2]
                    sentence = ' '.join(sentence)
                    #sentence = stemmer.stem_sentence(sentence)
                    #e1 = stemmer.stem_sentence(e1)
                    #e2 = stemmer.stem_sentence(e2)
                    s_unigrams = sentence.split()
                    feature.extend(s_unigrams)
                    feature.extend(ngramtolist(ngrams(s_unigrams,2)))
                    feature.extend(ngramtolist(ngrams(s_unigrams,2)))
                    feature.extend(ngramtolist(ngrams(s_unigrams,3)))
                    feature.extend(ngramtolist(ngrams(s_unigrams,3)))
                    feature.extend(ngramtolist(ngrams(s_unigrams,3)))
                    feature = ' '.join(feature)
                if r and count%4 == 1:
                    if r == 'Message-Topic(e1,e2)':
                        print(feature)
                    features.append(feature)
                    e1s.append(e1)
                    e2s.append(e2)
                    relations.append(r)
                count += 1
            data = pd.DataFrame({'feature':features, 'e1':e1s, 'e2':e2s, 'relation':relations})
        else:
            features,e1s,e2s,sids = [],[],[],[]
            for r in rows:
                if not r: break
                feature = []
                sid = r.split('\t')[0]
                sentence = r.split('\t')[1]
                sentence = s_rex.search(sentence).group(0)
                e1 = e1_rex.search(sentence).group(0)
                e2 = e2_rex.search(sentence).group(0)
                sentence = cleanhtml(sentence)
                for s in preps:
                    sentence = sentence.replace(s,s+' '+s)
                e1 = cleanhtml(e1)
                e2 = cleanhtml(e2)
                word_tokens = word_tokenize(sentence)
                sentence = [w.lower() for w in word_tokens if w.isalpha()]
                sentence = [lmtzr.lemmatize(i,'v') for i in sentence]
                e1 = [lmtzr.lemmatize(i,'v') for i in e1]
                e2 = [lmtzr.lemmatize(i,'v') for i in e2]
                sentence = ' '.join(sentence)
                #sentence = stemmer.stem_sentence(sentence)
                #e1 = stemmer.stem_sentence(e1)
                #e2 = stemmer.stem_sentence(e2)
                #dist = (sentence.index(e1)-sentence.index(e2))//5
                s_unigrams = sentence.split()
                feature.extend(s_unigrams)
                feature.extend(ngramtolist(ngrams(s_unigrams,2)))
                feature.extend(ngramtolist(ngrams(s_unigrams,2)))
                feature.extend(ngramtolist(ngrams(s_unigrams,3)))
                feature.extend(ngramtolist(ngrams(s_unigrams,3)))
                feature.extend(ngramtolist(ngrams(s_unigrams,3)))
                feature = ' '.join(feature)
                features.append(feature)
                e1s.append(e1)
                e2s.append(e2)
                sids.append(sid)
                count += 1
            data = pd.DataFrame({'feature':features, 'e1':e1s, 'e2':e2s, 'sid':sids})   
    return data

def training(x, y):
    print("Training...")
    print("Example: {}".format(x[0]))
    
    X_counts = count_vect.fit_transform(x)
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    print(X_tfidf.shape)
    model.fit(X_tfidf, y) 

def testing(x):
    print("Testing...")
    print("Example: {}".format(x[0]))
    X_counts = count_vect.transform(x)
    X_tfidf = tfidf_transformer.transform(X_counts)
    print(X_tfidf.shape)
    predicted = model.predict(X_tfidf)
    return predicted

def score(p):
    t = []
    with open("./answer_key.txt", "r") as f:
        rows = f.read().split('\n')
        for r in rows:
            if r:
                t.append(r.split('\t')[1])
    total = len(p)
    accuracy = sum([1 for i in range(total) if p[i]==t[i]])/total
    f1_micro = f1_score(t, p, average='micro')
    f1_macro = f1_score(t, p, average='macro')
    print("Accuracy = {:.4f}".format(accuracy))
    print("f1_micro = {:.4f}".format(f1_micro))
    print("f1_macro = {:.4f}".format(f1_macro))
    
def evaluate(t, p):
    total = len(p)
    accuracy = sum([1 for i in range(total) if p[i]==t[i]])/total
    f1_micro = f1_score(t, p, average='micro')
    f1_macro = f1_score(t, p, average='macro')
    print("Accuracy = {:.4f}".format(accuracy))
    print("f1_micro = {:.4f}".format(f1_micro))
    print("f1_macro = {:.4f}".format(f1_macro))

data = preprocessing(TRAINING_FILE_PATH)
training(data['feature'], data['relation'])
#predict = testing(data['feature'])
#evaluate(data['relation'],predict)
data = preprocessing(TESTING_FILE_PATH)
predicted = testing(data['feature'])
with open('./proposed_answer1.txt', 'w') as file:
    for i in range(len(predicted)):
        file.write(str(data.loc[i]['sid'])+'\t'+str(predicted[i])+'\n')
score(predicted)
