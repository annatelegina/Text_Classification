from sklearn.naive_bayes import  MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn import svm

#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
import pymorphy2

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import numpy as np
import argparse
import os

from data_loader import *

models = {
        "bayes": MultinomialNB, 
        "svm": svm.SVC
        }

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='./data', help="path to the dataset")
parser.add_argument("--model", default='bayes',choices=['bayes', 'svm'], help='type of model')
parser.add_argument("--stop_word", default=None)
parser.add_argument("--vectorization", default="freq", choices=["bool", "freq", "tfidf"], help='type of vectorization')
parser.add_argument('--max_features', default=500, type=int, help='number of features for vectorizing')

args = parser.parse_args()

def main():
    
    print("===> Preparing data...")
    data = TextDataLoader(path=args.dataset)
    corpus, target = data.get_data(mode='train')
    X_test, y_test = data.get_data(mode='test')

    if args.stop_word is not None:
        args.stop_word = stopwords.words('russian')

    if args.vectorization == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=args.max_features, stop_words=args.stop_word)
    else:
        vectorizer = CountVectorizer(max_features=args.max_features, stop_words=args.stop_word)

    a = []
    print(corpus[0])
    word_Lemmatized = pymorphy2.MorphAnalyzer()
    for i in corpus:
        i = word_tokenize(i)
        sen = []
        for word in i:
            word_f = word_Lemmatized.parse(word)[0].normal_form
            sen.append(word_f)
        a.append(concat_s(sen))

    corpus = a
    print(corpus[0])
    X_train = vectorizer.fit_transform(corpus).toarray()

    c = vectorizer.get_feature_names()
    print(len(c))
    X_test = vectorizer.transform(X_test).toarray()

    if args.vectorization == 'bool':
        X_train = X_train.astype(bool).astype(int)
        X_test = X_test.astype(bool).astype(int)

    print("===> Creating model...")
    model = models[args.model]()
    print("Model ", args.model, " was created")

    model.fit(X_train, target)

    prec, rec, f1, acc = evaluate(model, X_test, y_test)

    print("Results: Precision {:.3f}; Recall {:.3f}; F1-score: {:.3f}".format(prec, rec, f1))
    print("Accuracy: {:.3f}".format(acc))


def evaluate(model, test_data, target):
    out = model.predict(test_data)
    f1 = f1_score(target, out, average='weighted')
    prec = precision_score(target, out, average='weighted')
    rec = recall_score(target, out, average='weighted')
    acc = accuracy_score(target, out)
    return prec, rec, f1, acc

def concat_s(list):
    s = ""
    for i in list:
        s = s + " " +  i

    return s
if __name__ == "__main__":
    main()
