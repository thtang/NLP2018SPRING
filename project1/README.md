# Fine-Grained Sentiment Analysis on Financial Microblogs
## Task description
Given a text instance (microblog message about stock), predict the sentiment score.
## Dependency
`nltk` `xgboost` `afinn` `vaderSentiment` `scikit-learn` `numpy` 
## Usage
Training and testing:
```
python fine_grained.py [training json file ] [testing json file ]
```
For inference:
```
python inference.py [input string]
```
For example:
```
python inference.py "$TSLA borrow still -7.5% which means very tight on the avail stock for shorting....  $VALE meanwhile easing up to "only -8%""
```
Output:
```
fine-grained sentiment score: -0.106764615
```


# Sentiment Analysis F1 3-class

## Task description
Given a text instance (microblog message about stock), predict its sentiment class, i.e. bearish, neutral or bullish.

## Usage
```
python3 3class.py [input json file]
```
For example:
```
python3 3class.py test_set.json
```
Output:
```
Macro f1: 0.26
Micro f1: 0.70
```

## Reference
1. [NTUSD-Fin: A Market Sentiment Dictionary for Financial Social Media Data Applications](http://nlg3.csie.ntu.edu.tw/nlpresource/NTUSD-Fin/)
2. [Valence Aware Dictionary and sEntiment Reasoner](https://github.com/cjhutto/vaderSentiment)
3. [SemEval-2015 English Twitter Sentiment Lexicon](http://saifmohammad.com/WebPages/SCL.html#ETSL)
4. [AFINN sentiment analysis in Python](https://github.com/fnielsen/afinn)
