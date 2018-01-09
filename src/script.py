# basics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# preprocessing and streams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import re
import unidecode
# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
# local imports
import czech_stemmer as stem
import weight_averaging


REGEX = re.compile('[^a-zA-Z]')
NEGATIVE_DATA_PATH = '../data/negative.txt'
POSITIVE_DATA_PATH = '../data/positive.txt'
NEUTRAL_DATA_PATH = '../data/neutral.txt'
STOPWORDS_PATH = '../data/stopwords.txt'
sizes_pos = []
sizes_neg = []
sizes_neu = []

# load data from txt files except stopwords
with open(STOPWORDS_PATH, encoding="utf8") as f:
    stopwords = f.readlines()
with open(NEGATIVE_DATA_PATH, encoding="utf8") as f:
    neg_sens = f.readlines()
with open(POSITIVE_DATA_PATH, encoding="utf8") as f:
    pos_sens = f.readlines()
with open(NEUTRAL_DATA_PATH, encoding="utf8") as f:
    neu_sens = f.readlines()

# preprocess string    
def to_string(str):
    # remove accents and lower case
    without_accents = unidecode.unidecode(str).lower()
    # keep only a-z
    without_accents = REGEX.sub('', without_accents)
    return without_accents

# check if is string among stopwords if yes returns empty string
# otherwise returns string
def remove_stopwords(str):
    # if is str in our stopwords return '' otherwise return str
    if any(str in s for s in stopwords):
        return ''
    return str

# create concatation of data word by word
def concat_data(X_train_data, X_train_labels, loaded_sentences, label):
    flatten_list = flatten(loaded_sentences)
    for i in range(len(flatten_list)):
        X_train_data.append(flatten_list[i])
        X_train_labels.append(label)    

# create flatten representation of data        
def flatten(loaded_sentences):
    flatten_sentence = []
    for i in range(len(loaded_sentences)):
        for j in range(len(loaded_sentences[i])):
            flatten_sentence.append(loaded_sentences[i][j])
    return flatten_sentence

# build sentence and compute lengths of sentences according to label
def build_sentences(X_data, Y_data, lists, label):
    for i in range(len(lists)):
        sentence = ''
        if label == 'pos':
            sizes_pos.append(len(lists[i]))
        if label == 'neu':
            sizes_neu.append(len(lists[i]))
        if label == 'neg':
            sizes_neg.append(len(lists[i]))
        for j in range (len(lists[i])):
            sentence += lists[i][j] + ' '
        X_data.append(sentence)
        Y_data.append(label)

# expected list of sentences which tokenize
# preprocess and returns
def preprocess_sentences(loaded_sentences):
    sentences = []
    #for i in range(len(loaded_sentences)):
    # LOAD ONLY 100 TEXTS FOR TESTING PURPOSE
    for i in range(len(loaded_sentences)):
    #for i in range(1000):
        splitted = loaded_sentences[i].split()
        preprocessed_sentence = []
        for j in range(len(splitted)):
            formatted = to_string(splitted[j])
            without_stopwords = remove_stopwords(formatted)
            if (without_stopwords == ''):
                continue
            # stemming can be agressive or not according to flag
            stemmed = stem.cz_stem(without_stopwords)
            preprocessed_sentence.append(stemmed)
        sentences.append(preprocessed_sentence)
    return sentences

# for creating categorized data
def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded

######################## MAIN

print ('#### Preprocessing started')
# preprocess sentences
prep_pos_sens = preprocess_sentences(pos_sens)
prep_neu_sens = preprocess_sentences(neg_sens)
prep_neg_sens = preprocess_sentences(neu_sens)

# divide into testing and training set
train_pos = prep_pos_sens[0:int(len(prep_pos_sens) * 0.9)]
test_pos = prep_pos_sens[int(len(prep_pos_sens) * 0.9) : ]
train_neu = prep_neu_sens[0:int(len(prep_neu_sens) * 0.9)]
test_neu = prep_neu_sens[int(len(prep_neu_sens) * 0.9) : ]
train_neg = prep_neg_sens[0:int(len(prep_neg_sens) * 0.9)]
test_neg = prep_neg_sens[int(len(prep_neg_sens) * 0.9) : ]

# create train set
X_train = []
Y_train = []
build_sentences(X_train, Y_train, train_pos, 'pos')
build_sentences(X_train, Y_train, train_neu, 'neu')
build_sentences(X_train, Y_train, train_neg, 'neg')
# create test set
X_test = []
Y_test = []
build_sentences(X_test, Y_test, test_pos, 'pos')
build_sentences(X_test, Y_test, test_neu, 'neu')
build_sentences(X_test, Y_test, test_neg, 'neg')

'''
# Print mean of sentence lengths according to label
a = np.median(np.array(sizes_pos))
print('sizes_pos = ' + str(a))
b = np.median(np.array(sizes_neg))
print('sizes_neg = ' + str(b))
c = np.median(np.array(sizes_neu))
print('sizes_neu = ' + str(c))
'''

print ('#### Preprocessing done')

# build models
print ('#### Building models started')

# this is not best solution but for simplify model averaging we will
# use CountVectorizer and TfidfTransformer for 3 times on same data
pipeline_nb = Pipeline([
    ('vect', CountVectorizer(lowercase=False, max_df = 0.8, max_features = 50000, ngram_range = (1, 3))),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])
    
pipeline_sgd = Pipeline([
    ('vect', CountVectorizer(lowercase=False, max_df = 0.8, max_features = 50000, ngram_range = (1, 3))),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier())
])
    
pipeline_lr = Pipeline([
    ('vect', CountVectorizer(lowercase=False, max_df = 0.8, max_features = 50000, ngram_range = (1, 3))),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression())
])
    
print ('#### Building models done')
  
'''
print ('#### Grid search started')

# Set parameters for grid search
parameters = {
    'vect__max_df': (0.7, 0.8, 0.9),
    #'vect__max_features': (5000, 10000, 25000, 50000),
    'vect__ngram_range': ((1, 2), (1, 3), (1, 4), (1, 5)),
    #'tfidf__use_idf': (True, False),
    #'clf__n_iter': (10, 50, 80)
}

# Use GridSearch and Cross-Validation
grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1)
grid_search.fit(X_train, Y_train)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

print ('#### Grid search done')
'''    

print ('#### Classification started')

pipeline_nb.fit(X_train, Y_train)
Y_pred_nb = pipeline_nb.predict(X_test)
acc_nb = round(pipeline_nb.score(X_test, Y_test) * 100, 2)

pipeline_sgd.fit(X_train, Y_train)
Y_pred_sgd = pipeline_sgd.predict(X_test)
acc_sgd = round(pipeline_sgd.score(X_test, Y_test) * 100, 2)

pipeline_lr.fit(X_train, Y_train)
Y_pred_lr = pipeline_lr.predict(X_test)
acc_lr = round(pipeline_lr.score(X_test, Y_test) * 100, 2)

print ('Classification done')

# Print correlation matrix of results
predictions = pd.DataFrame({
      'Logistic Regression': Y_pred_lr.ravel(), 
      'Multinomial Naive Bayes': Y_pred_nb.ravel(),
      'Stochastic Gradient Decent': Y_pred_sgd.ravel()                                         
})
predictions['Logistic Regression'] = coding(predictions['Logistic Regression'], {'pos':0,'neu':1,'neg':2})
predictions['Multinomial Naive Bayes'] = coding(predictions['Multinomial Naive Bayes'], {'pos':0,'neu':1,'neg':2})
predictions['Stochastic Gradient Decent'] = coding(predictions['Stochastic Gradient Decent'], {'pos':0,'neu':1,'neg':2})
corrmat = predictions.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True, annot=True)


# Unfornutally model averaging does not help - methods are too correlated
# but I want to show this approach
print('#### Model averaging started')

weights = [1, 1, 1]
results = [Y_pred_nb, Y_pred_sgd, Y_pred_lr]
res_average = weight_averaging.average_weight(results, weights)

# compute finale score
counter = 0
for i in range(len(Y_test)):
    if Y_test[i] == res_average[i]:
        counter += 1
        
score = counter / len(Y_test)

print('#### Model averaging done')

print("---- Final score: %0.3f" % score)
