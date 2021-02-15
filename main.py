import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from utils import *
from preprocessing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from joblib import load, dump
from sklearn.metrics import f1_score, precision_score, recall_score
from joblib import load, dump
import sys

trainingDataFile = "reviews_train.csv"
testDataFile = "reviews_test.csv"
min_word_count = 5

def prepareTrainData(filePath):
    print("PreparingData...")
    reviews = importData(trainingDataFile)
    reviews = reviews[["score", "reviewText"]]
    reviews = reviews.dropna()
    reviews = reviews.sample(frac=0.1)
    common_words = readCommonWords()
    feature_dict = prepareCommonWordsDictionary(common_words)

    return create_bow(reviews, feature_dict) # X_train, y_train

def saveMostCommonWords(texts, min_count):
    print("Finding most common words...")
    most_common = getUniqueWords(texts, min_count)
    writeCommonWordsToFile(most_common)

def train(X_train, y_train):
    trainRandomForest(X_train, y_train)
    trainDummy(X_train, y_train)
    trainLinearSVC(X_train, y_train)
    trainMultinomialNB(X_train, y_train)

def trainRandomForest(X_train, y_train):
    print("Training RandomForest...")
    random_forest = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=23)
    random_forest.fit(X_train, y_train)
    dump(random_forest, "classifiers/random_forest.joblib")

def trainDummy(X_train, y_train):
    print("Training Dummy...")
    dummy = DummyClassifier(strategy = 'most_frequent')
    dummy.fit(X_train, y_train)
    dump(dummy, 'classifiers/dummy.joblib')

def trainLinearSVC(X_train, y_train):
    print("Training LinearSVC...")
    svm = LinearSVC(class_weight='balanced')
    svm.fit(X_train, y_train)
    dump(svm, 'classifiers/svm.joblib')

def trainMultinomialNB(X_train, y_train):
    print("Traning MultinominalNB...")
    nbc = MultinomialNB()
    nbc.fit(X_train, y_train)
    dump(nbc, 'classifiers/nbc.joblib')

def loadDummy():
    return load('classifiers/dummy.joblib')

def loadRandomForest():
    return load("classifiers/random_forest.joblib")

def loadSVM():
    return load("classifiers/svm.joblib")

def loadNBC():
    return load("classifiers/nbc.joblib")

def testingData(X, y, classifier):
    y_pred = classifier.predict(X)
    print("=================== Results ===================")
    print("Classifier:", type(classifier))
    print("            1.0    2.0    3.0    4.0     5.0")
    print("F1       ", np.round(f1_score(y, y_pred, average=None, pos_label=None, labels=np.unique(y)), 4))
    print("Precision", np.round(precision_score(y, y_pred, average=None, pos_label=None, labels=np.unique(y)), 4))
    print("Recall   ", np.round(recall_score(y, y_pred, average=None, pos_label=None, labels=np.unique(y)), 4), "\n")

if __name__ == '__main__':
    # Training Classifiers
    # X_train, y_train = prepareTrainData(trainingDataFile)
    # train(X_train, y_train)

    # Training Single Classfier
    # trainRandomForest(X_train, y_train)
    #
    # # Testing Classifiers
    print("Loading testing data...")
    reviews = importData(testDataFile)
    reviews = reviews[["score", "reviewText"]]
    reviews = reviews.dropna()
    reviews = reviews.sample(frac=0.5)

    # # scoreHistogram(reviews)
    print("Loading common words...")
    common_words = readCommonWords()

    feature_dict = prepareCommonWordsDictionary(common_words)

    print("Preparing data for classifiers..")
    X, y = create_bow(reviews, feature_dict)

    testingData(X, y, loadSVM())
    testingData(X, y, loadDummy())
    testingData(X, y, loadNBC())
    testingData(X, y, loadRandomForest())
