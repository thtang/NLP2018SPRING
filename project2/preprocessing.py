# -*- coding: utf-8 -*-
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle as pk
import re
from bs4 import BeautifulSoup
import spacy
import sys
def load_data(FILE_Path):
    otuput_sent = []
    lemmatizer = WordNetLemmatizer()
    with open(FILE_Path, "r") as f:
        train_tmp = f.read().split("\n\n")[:-1]
        for sample in train_tmp:
            sentence = re.split('\t|\n', sample)[1].replace('\"',"").replace(".","").replace(",","")            .replace("?","").replace("!","").replace(")","").replace("(","").replace("\'s","")            .replace("\'ve","").replace("\'t","").replace("\'re","").replace("\'d","")            .replace("\'ll","").replace("'","").replace(";","").replace(":","")
            answer = re.split('\t|\n', sample)[2]
            soup = BeautifulSoup(sentence,"lxml")

            e1 = str(soup.find('e1'))[4:-5]
            e2 = str(soup.find('e2'))[4:-5]
            word_list = sentence.split(" ")
            e1_check = 0
            e2_check = 0
            for word in word_list:
                if word.endswith("</e1>"):
                    e1_check = 1
                if word.endswith("</e2>"):
                    e2_check = 1
            if e1_check + e2_check != 2:
                print(sentence)
            for word in word_list:
                if "</e1>" in word:
                    a = word
                if "</e2>" in word:
                    b = word
            p1 = word_list.index(a)
            p2 = word_list.index(b)
            sentence = sentence.replace("<e1>","").replace("</e1>","").replace("<e2>","").            replace("</e2>","")
            sentence = [lemmatizer.lemmatize(word, pos='v') for word in sentence.split(" ")]
            sentence = " ".join(sentence)
            otuput_sent.append((sentence,e1,e2,p1,p2,answer))
    return otuput_sent


#################################################
#Load training and Testing data 
#################################################
#sys.argv[1] = ./data/TRAIN_FILE.txt
#sys.argv[2] = ./data/TEST_FILE_FULL.txt
train_instance = load_data(sys.argv[1])
test_instance = load_data(sys.argv[2])
total = train_instance + test_instance


#################################################
#Depency parsing features 
#################################################
nlp = spacy.load('en_core_web_sm')
train_text = [ele[0] for ele in train_instance]
test_test = [ele[0] for ele in test_instance]
train_mut_ancestors_list = []
train_dep_list = []
for ele in train_instance:
    sentence = ele[0]
    doc = nlp(sentence)
    entity = ele[1:3]
    both_ancestors = []
    both_dep = ""
    for pos, token in enumerate(doc):
        if pos in ele[3:5]:
            ancestors = [i.text for i in token.ancestors]
            both_ancestors.append(ancestors)
            both_dep+=token.dep_
    a = 0
    b = 0
    if entity[0] in both_ancestors[1]:
        a = 1
    elif entity[1] in both_ancestors[0]:
        b = 1
    mut_ancestors = [a,b]
    train_mut_ancestors_list.append(mut_ancestors)
    train_dep_list.append(both_dep)
    
test_mut_ancestors_list = []
test_dep_list = []
for ele in test_instance:
    sentence = ele[0]
    doc = nlp(sentence)
    entity = ele[1:3]
    both_ancestors = []
    both_dep = ""
    for pos, token in enumerate(doc):
        if pos in ele[3:5]:
            ancestors = [i.text for i in token.ancestors]
            both_ancestors.append(ancestors)
            both_dep+=token.dep_
    a = 0
    b = 0
    if entity[0] in both_ancestors[1]:
        a = 1
    elif entity[1] in both_ancestors[0]:
        b = 1
    mut_ancestors = [a,b]
    test_mut_ancestors_list.append(mut_ancestors)
    test_dep_list.append(both_dep)

np.save("./features/train_mut_ancestors_list.npy", train_mut_ancestors_list)
np.save("./features/test_mut_ancestors_list.npy", test_mut_ancestors_list)


a, b = np.unique(train_dep_list, return_counts=True)
a_sorted = a[np.argsort(b)[::-1]]
major_dep = a_sorted[:33]

train_dep_list_filter = []
for dep in train_dep_list:
    if dep in major_dep:
        train_dep_list_filter.append(dep)
    else:
        train_dep_list_filter.append("other")

test_dep_list_filter = []
for dep in test_dep_list:
    if dep in major_dep:
        test_dep_list_filter.append(dep)
    else:
        test_dep_list_filter.append("other")

np.unique(test_dep_list_filter, return_counts=True)



le = LabelEncoder()
enc = OneHotEncoder(sparse=False)
lb = LabelBinarizer()

# transform string to onehot
train_dep_list_filter = lb.fit_transform(train_dep_list_filter)
test_dep_list_filter = lb.transform(test_dep_list_filter)

np.save("./features/train_dep_list.npy", train_dep_list_filter)
np.save("./features/test_dep_list.npy", test_dep_list_filter)


#################################################
#WordNet hypernym features 
#################################################
hyper_list = []
for idx in range(len(total)):
    entity_1 = total[idx][1]
    entity_2 = total[idx][2]
    
    synsets = wn.synsets(entity_1, pos=wn.NOUN)
    hyper_list_1 = []
    for syn in synsets:
        for hyper in syn.hypernyms():
            hyper_list_1 = hyper_list_1 + [x.lower() for x in hyper.lemma_names() if '_' not in x]
    
    synsets = wn.synsets(entity_2, pos=wn.NOUN)
    hyper_list_2 = []
    for syn in synsets:
        for hyper in syn.hypernyms():
            hyper_list_2 = hyper_list_2 + [x.lower() for x in hyper.lemma_names() if '_' not in x]
    hyper = ' '.join(list(set(hyper_list_1).union(set(hyper_list_2)))) 
    hyper_list.append(hyper)

train_hyper_list = hyper_list[:8000]
test_hyper_list = hyper_list[8000:]



np.save("./features/train_hyper_list.npy", train_hyper_list)
np.save("./features/test_hyper_list.npy", test_hyper_list)



#################################################
#PropBank features 
#################################################
# propBank_dict parsing from google SLING result
with open('dict/propBank_dict.pk', 'rb') as f:
    propBank_dict = pk.load(f)

lmtzr = WordNetLemmatizer()
propBank_verb_entity_dict = {}
hyper_verb_list = []
for idx in range(len(total)):
    prop_rule = []
    temp_dict = {}
    for prop_rule in propBank_dict[idx]['propBank']:
        verb = prop_rule[0].split('.')[0]
        verb = lmtzr.lemmatize(verb,'v')
        entity = prop_rule[4]
        if len(entity.split(' ')) > 1:
            entity = entity.split(' ')[-1]
        #print(entity)
        if verb in temp_dict:
            temp_dict[verb].append(entity)
        else:
            temp_dict[verb] = [entity]
    
    temp_hyper_list = []
    for k, v in temp_dict.items():
        if total[idx][1] in v and total[idx][2] in v:
            synsets = wn.synsets(k, pos=wn.VERB)
            temp_hyper_list.append(k)

            for syn in synsets:
                if len(syn.hypernyms()) != 0:
                    temp_hyper_list = temp_hyper_list + [x.lower() for x in syn.hypernyms()[0].lemma_names() if '_' not in x]
            
    hyper = ' '.join(list(set(temp_hyper_list)))
    hyper_verb_list.append(hyper)

train_hyper_verb_list = hyper_verb_list[:8000]
test_hyper_verb_list = hyper_verb_list[8000:]

np.save("./features/train_hyper_verb_list.npy", train_hyper_verb_list)
np.save("./features/test_hyper_verb_list.npy", test_hyper_verb_list)