import re
import nltk
import pickle as pk


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def labels_mapping(relation):
    labelsMapping = {'Other': 0,
                 'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
                 'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
                 'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
                 'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                 'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
                 'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
                 'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
                 'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
                 'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}
    return labelsMapping[relation]

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end].strip()
    except ValueError:
        return ""

def read_data(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        #lines = lines[:10]
    return lines

def load_train_data(path):
    data = []
    sentence_list = []
    data_label = []
    position = []
    lines = read_data(path)     
    for idx in range(0, len(lines), 4):
        ID = lines[idx].split("\t")[0]
        
        sentence = lines[idx].split("\t")[1][1:-1]
        sentence_list.append(sentence.replace('<e1>', "").replace('<e2>', "").replace('</e1>', "").replace('</e2>', ""))
        sentence = clean_str(sentence)
        entity_1 = find_between(sentence, 'e1', 'e1')
        entity_2 = find_between(sentence, 'e2', 'e2')

        if len(entity_1.split(' ')) > 1:
            entity_1 = entity_1.split(' ')[-1]
        if len(entity_2.split(' ')) > 1:
            entity_2 = entity_2.split(' ')[-1]
        sentence = sentence.replace('e1 ', "").replace(' e2', "").replace('e2 ', "").strip()

        p1 = sentence.split(' ').index(entity_1)
        p2 = sentence.split(' ').index(entity_2)
        if p1 == p2:
            p2 = [index for index in range(len(sentence.split(' '))) if sentence.split(' ')[index] == entity_2][1]
        position.append([p1,p2])
        
        relation = lines[idx + 1]
        relation_label = labels_mapping(relation)
        
        data.append([ID,sentence, entity_1, entity_2])
        data_label.append([ID, relation, relation_label])
        
    return data, data_label, position, sentence_list

def load_test_data(path):
    data = []
    position = []
    sentence_list = []
    lines = read_data(path)
    for idx in range(0, len(lines)):
        ID = lines[idx].split("\t")[0]
        
        sentence = lines[idx].split("\t")[1][1:-1]
        sentence_list.append(sentence.replace('<e1>', "").replace('<e2>', "").replace('</e1>', "").replace('</e2>', ""))
        sentence = clean_str(sentence)
        entity_1 = find_between(sentence, 'e1', 'e1')
        entity_2 = find_between(sentence, 'e2', 'e2')
        if len(entity_1.split(' ')) > 1:
            entity_1 = entity_1.split(' ')[-1]
        if len(entity_2.split(' ')) > 1:
            entity_2 = entity_2.split(' ')[-1]
        sentence = sentence.replace('e1 ', "").replace(' e2', "").replace('e2 ', "").strip()
        
        p1 = sentence.split(' ').index(entity_1)
        p2 = sentence.split(' ').index(entity_2)
        if p1 == p2:
            p2 = [index for index in range(len(sentence.split(' '))) if sentence.split(' ')[index] == entity_2][1]
        position.append([p1,p2])
        data.append([ID, sentence, entity_1, entity_2])
    return data, position, sentence_list

def load_test_answer(path):
    lines = read_data(path)
    data = []
    for idx in range(0, len(lines)):
        ID = lines[idx].split("\t")[0]
        relation = lines[idx].split("\t")[1]
        relation_label = labels_mapping(relation)
        data.append([ID, relation, relation_label])
    return data

if __name__ == '__main__':
    train, train_label, train_position, train_sentence = load_train_data('data/TRAIN_FILE.txt')
    test, test_position, test_sentence = load_test_data('data/TEST_FILE.txt')
    test_label = load_test_answer('data/answer_key.txt')
    print("number of training instances:", len(train))
    print("number of testing instances:", len(test)) 