import sys
import json
import numpy as np
import gensim
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Embedding,Input,BatchNormalization,Dense,Bidirectional,LSTM,Dropout
from keras.callbacks import History ,ModelCheckpoint, EarlyStopping

input_filepath = sys.argv[1]

NUM_CLASS = 3
CLASS_THRESHOLD = 0.1

#################### Preprocessing Tweet
def preprocess(data):
    text = data["tweet"]
    
    #Word Level Preprocessing
    tokens = text.split(' ')
    tokens = [t for t in tokens if not (t.find("http")>=0 or t.find('$')>=0 or t.find('@')>=0 or t.find('/')>=0)]
    text = ' '.join(t for t in tokens if t not in stopwords)
    
    #Char Level Preprocessing
    tokens = [c for c in text if c not in stopwords]
    text = ''.join(tokens)
    text = text.lower()

    return text

#################### Calculate micro-f1 and macro-f1
def mxcrof1(y_true, y_pred, mode="micro"):
    tps = [0] * NUM_CLASS
    fps = [0] * NUM_CLASS
    fns = [0] * NUM_CLASS
    gt =np.argmax(y_true, axis=1)
    pred = np.argmax(y_pred, axis=1)
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            tps[gt[i]] += 1
        else:
            fns[gt[i]] += 1
            fps[pred[i]] += 1   
    if mode == "macro":
        precisions = [x/(x+y) if (x+y)>0 else 0 for x, y in zip(tps, fps)]
        recalls = [x/(x+y) if (x+y)>0 else 0 for x, y in zip(tps, fns)]
        avg_precision = sum(precisions)/NUM_CLASS
        avg_recall = sum(recalls)/NUM_CLASS     
    else:
        sum_tps = sum(tps)
        sum_fps = sum(fps)
        sum_fns = sum(fns)
        avg_precision = sum_tps/(sum_tps+sum_fps) if (sum_tps+sum_fps)>0 else 0
        avg_recall = sum_tps/(sum_tps+sum_fns) if (sum_tps+sum_fns)>0 else 0
    return 2*(avg_precision*avg_recall)/(avg_precision+avg_recall)

#################### Read Training Data from: ./training_set.json
train_origin = []
with open(input_filepath, encoding='utf8') as f:
    train_json = json.load(f)
    for data in train_json:
        tweet = data['tweet']
        target = data['target']
        snippet = data['snippet']
        sentiment = data['sentiment']
        train_origin.append({'tweet':tweet, 'target':target, 'snippet':snippet, 'sentiment':sentiment})

#################### Read Stopwords from: ./stopwords.txt
stopwords = []
with open("./3class-stopwords.txt", "r") as f:
    stopwords = f.read().split('\n')

#################### Stemming
stemmer = gensim.parsing.porter.PorterStemmer()
train_stemmed = []
for data in train_origin:
    cleaned = preprocess(data)
    train_stemmed.append(stemmer.stem_sentence(cleaned))

train_corpus = [sent.split(" ") for sent in train_stemmed]

#################### Training/Loading W2V model
emb_size = 300
#w2v_model = gensim.models.Word2Vec(train_corpus, size=emb_size, window=5, min_count=0)
#w2v_model.save("modelW2V")
w2v_model = gensim.models.Word2Vec.load("3class-model")

vocab_size = None
tokenizer = Tokenizer(num_words=vocab_size,filters="\n\t")
tokenizer.fit_on_texts(train_stemmed)
sequences = tokenizer.texts_to_sequences(train_stemmed)
word_index = tokenizer.word_index

oov_count = 0
embedding_matrix = np.zeros((len(word_index)+1, emb_size))
for word, i in word_index.items():
    try:
        embedding_vector = w2v_model.wv[word]
        embedding_matrix[i] = embedding_vector
    except:
        oov_count +=1
        
max_length = np.max([len(i) for i in sequences])
print("Max Sentence Length: {}".format(max_length))

train_X = pad_sequences(sequences, maxlen=max_length)

#################### 3-Class: 0:Bullish, 1:Bearish, 2:Neutral
train_Y = []
for data in train_origin:
    sentiment = float(data['sentiment'])
    if sentiment > CLASS_THRESHOLD:
        train_Y.append(0)
    elif sentiment < -CLASS_THRESHOLD:
        train_Y.append(1)
    else:
        train_Y.append(2)
train_Y = to_categorical(train_Y)

#################### Keras LSTM net with embedding layer
print("Dimension: {}".format(emb_size))
print("Vocab Size: {}".format(len(w2v_model.wv.vocab)))
print("Embedding Shape: {}".format(embedding_matrix.shape))
embedding_layer = Embedding(len(word_index)+1,output_dim= emb_size,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)
batch_size = 64
attempts = 1
for i in range(attempts):
    train_x, valid_x, train_y, valid_y = train_test_split(train_X, train_Y, test_size=0.2)
    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    lstm1 = Bidirectional(LSTM(128,activation="tanh",dropout=0.2,return_sequences = True))(embedded_sequences)
    lstm2 = Bidirectional(LSTM(64,activation="tanh",dropout=0.2,return_sequences = False))(lstm1)
    bn1 = BatchNormalization()(lstm2)
    dense1 = Dense(64, activation="relu")(bn1)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(32, activation="relu")(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    preds = Dense(3, activation='softmax')(dropout2)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    hist = History()
    check_save  = ModelCheckpoint("./models/modelW2V-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)
    early_stop = EarlyStopping(monitor="val_loss", patience=2)
    #model.summary()
    #from keras.utils import plot_model 
    #plot_model(model, to_file='model.png')
    model.fit(train_x, train_y, validation_data=(valid_x, valid_y),epochs=10, batch_size=batch_size,callbacks=[check_save,hist,early_stop])
    result = model.predict(valid_x, batch_size=batch_size, verbose=True)
    print("Macro f1: {}".format(mxcrof1(valid_y, result, "macro")))
    print("Micro f1: {}".format(mxcrof1(valid_y, result, "micro")))