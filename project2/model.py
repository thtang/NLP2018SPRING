# -*- coding: utf-8 -*-
import re
import sys
import numpy as np
from bs4 import BeautifulSoup
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.callbacks import History 
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Embedding, Input,BatchNormalization, Dense, Bidirectional,LSTM,Dropout,Activation, concatenate
from keras import backend as K
LabelsMapping = {'Other': 0,
                 'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
                 'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
                 'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
                 'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                 'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
                 'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
                 'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
                 'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
                 'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}

def _shuffle(X, feature1, feature2, y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], feature1[randomize], feature2[randomize], y[randomize])

def load_data(FILE_Path):
    otuput_sent = []
    with open(FILE_Path, "r") as f:
        train_tmp = f.read().split("\n\n")[:-1]
        for sample in train_tmp:
            sentence = re.split('\t|\n', sample)[1].replace('\"',"").replace(".","").replace(",","")            .replace("?","").replace("!","").replace(")","").replace("(","").replace("\'s","")            .replace("\'ve","").replace("\'t","").replace("\'re","").replace("\'d","").replace("\'ll","")
            answer = re.split('\t|\n', sample)[2]
            soup = BeautifulSoup(sentence,"lxml")

            e1 = str(soup.find('e1'))[4:-5]
            e2 = str(soup.find('e2'))[4:-5]
            word_list = sentence.split(" ")
            for word in word_list:
                if "</e1>" in word:
                    a = word
                if "</e2>" in word:
                    b = word
            p1 = word_list.index(a)
            p2 = word_list.index(b)
            sentence = sentence.replace("<e1>","").replace("</e1>","").replace("<e2>","").            replace("</e2>","")

            otuput_sent.append((sentence,e1,e2,p1,p2,answer))
    return otuput_sent



train_instance = load_data(sys.argv[1])
test_instance = load_data(sys.argv[2])
# sentence_list = train_sentence + test_sentence
print("number of training instances:", len(train_instance))
print("number of testing instances:", len(test_instance))
# tuple format: (text, e1, e2, pos1, pos2, answer)

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

num_classes = 19
train_text = []
for instance in train_instance:
    p1 = instance[3]
    p2 = instance[4]
    split_sentence = instance[0].split(" ")
    p1_mod = max(0,p1-3)
    p2_mod = min(len(split_sentence), p2+3)
    prune_text = " ".join(split_sentence[p1:p2+1])
    train_text.append(prune_text)

train_label = [LabelsMapping[ele[5]] for ele in train_instance]
train_label = dense_to_one_hot(np.array(train_label), 19)

test_text = []
for instance in test_instance:
    p1 = instance[3]
    p2 = instance[4]
    split_sentence = instance[0].split(" ")
    p1_mod = max(0,p1-3)
    p2_mod = min(len(split_sentence), p2+3)
    prune_text = " ".join(split_sentence[p1:p2+1])
    test_text.append(prune_text)
    
test_label = [LabelsMapping[ele[5]] for ele in test_instance]
test_label = dense_to_one_hot(np.array(test_label), 19)

total_text = train_text + test_text

#################################################
#Load features
#################################################
train_mut_ancestors_list = np.load("./features/train_mut_ancestors_list.npy")
test_mut_ancestors_list = np.load("./features/test_mut_ancestors_list.npy")
train_dep_list = np.load("./features/train_dep_list.npy")
test_dep_list = np.load("./features/test_dep_list.npy")

#################################################
#Tokenizer
#################################################
tokenizer = Tokenizer(num_words=25000,lower=True,split=' ',char_level=False)
tokenizer.fit_on_texts(total_text)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

train_sentence_seq = tokenizer.texts_to_sequences(train_text)
test_sentence_seq = tokenizer.texts_to_sequences(test_text)
max_length = np.max([len(i) for i in train_sentence_seq+test_sentence_seq])
print("max length:", max_length)

x_train_seq = sequence.pad_sequences(train_sentence_seq, maxlen=max_length)
x_test_seq = sequence.pad_sequences(test_sentence_seq, maxlen=max_length)


#################################################
#Build embedding_matrix
#################################################
# download pre-trained word vector from "https://nlp.stanford.edu/projects/glove/"
tmp_file = get_tmpfile(sys.argv[3])
w2vModel = KeyedVectors.load_word2vec_format(tmp_file)


#################################################
#prepare embedding matrix
#################################################
embedding_size = 300
num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words, embedding_size))
oov = 0
for word, i in word_index.items():
    if word in w2vModel.wv.vocab:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = w2vModel[word]
    else:
        oov+=1
print("OOV:",oov)



#################################################
#Training
#################################################
def swish(x):
    return (K.sigmoid(x) * x)
get_custom_objects().update({'swish': Activation(swish)})

def train_BiLSTM(x_train, ancestor_train, dep_train, y_train, 
                 x_val, ancestor_val, dep_val, y_val,
                 embedding_matrix, max_length, max_features):
    embedding_size = 300
    batch_size = 64
    epochs = 50
    embedding_layer = Embedding(max_features,output_dim= embedding_size,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)
    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    lstm0 = Bidirectional(LSTM(256,activation="tanh",dropout=0.2,return_sequences = True,
                kernel_initializer='he_uniform'))(embedded_sequences)
    lstm1 = Bidirectional(LSTM(128,activation="tanh",dropout=0.2,return_sequences = True,
                kernel_initializer='he_uniform'))(lstm0)
    lstm2 = Bidirectional(LSTM(64,activation="tanh",dropout=0.2,return_sequences = False,
                kernel_initializer='he_uniform'))(lstm1)
    bn1 = BatchNormalization()(lstm2)
    
    # other feature inputs 
    ancestor_input = Input(shape=(2,))
    ancestor_feature = Dense(64, activation=swish)(ancestor_input)
    
    
    dep_input = Input(shape=(34,))
    dep_feature = Dense(128, activation=swish)(dep_input)
    
    combine_feature = concatenate([bn1, ancestor_feature, dep_feature])
    dense1 = Dense(64, activation=swish)(combine_feature)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(32, activation=swish)(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    preds = Dense(19, activation='softmax')(dropout2)
    model = Model([sequence_input, ancestor_input, dep_input], preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    filepath = "./BiLSTM_3.hdf5" 
    checkpoint = ModelCheckpoint(filepath,monitor='val_acc',save_best_only=True)
    history = History()
    callbacks_list = [checkpoint, history]
    
    history = model.fit([x_train, ancestor_train, dep_train], y_train, 
                        validation_data=([x_val, ancestor_val, dep_val], y_val), 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=callbacks_list)
    return model, history


train_X, train_ancestor, train_dep, train_y = _shuffle(x_train_seq, 
                                                       train_mut_ancestors_list,
                                                       train_dep_list,
                                                       train_label)
model, history = train_BiLSTM(train_X, train_ancestor, train_dep, train_y, 
                     x_test_seq, test_mut_ancestors_list, test_dep_list, test_label,
                     embedding_matrix,
                     max_length,
                     len(word_index)+1)


#################################################
#Testing
#################################################
y_test = [np.where(r==1)[0][0] for r in test_label ]
prediction = model.predict([x_test_seq,test_mut_ancestors_list,test_dep_list], batch_size=1000)
pred_y = np.argmax(prediction,axis=1)


LabelsMapping_inv =  {v: k for k, v in LabelsMapping.items()}
test_id = list(range(8001,8001+len(pred_y)))

with open(sys.argv[4], "w") as f:
    for i in range(len(test_id)):
        f.write(str(test_id[i])+"\t"+pred_y[i])
        f.write("\n")