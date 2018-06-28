# SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals

## Task description
The task was designed to compare different approaches to semantic relation classification and to provide a standard testbed for future research.
## Dependency
`nltk` `spacy` `keras` `vaderSentiment` `scikit-learn` `numpy` `wordNet` `propBank`
## Usage
Generate training and testing features:
```
python preprocessing.py [training data path] [testing data path]
```
For training and testing:<br>
* Download the pre-trained word vector [[link]](https://goo.gl/uvoCoK).
```
python train.py [training data path] [testing data path] [pertrain w2v] [result file path]
```
For evaluation:
```
perl semeval2010_task8_scorer-v1.2.pl [result file path] [answer_key file path] > [result score file path]
```
## Result  
<img src="https://github.com/thtang/NLP/blob/master/project2/pic/result_f1%20score.PNG" width=620>

## Reference
1. [SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals](http://www.aclweb.org/anthology/S10-1006)
2. [UTD: Classifying Semantic Relations by Combining Lexical and Semantic Resources](http://www.aclweb.org/anthology/S10-1057)
3. [SLING - A natural language frame semantics parser](https://github.com/google/sling)
