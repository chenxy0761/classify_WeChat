# -*- coding:UTF-8 -*-

from __future__ import print_function
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import time as tm
import matplotlib.pyplot as plt
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import random
import cPickle

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")
(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

###############################################################################
#labeldict = { u"是":1,u"否":0 }
categories = []
print('data loaded')
path = u"大宗商品seg.txt"
outputfile = "dzsptfidf"
if not os.path.exists(outputfile):
    os.mkdir(outputfile)

imagepath =  os.path.join(outputfile,"perf.png")
#outputfile = u"E://项目需求//JDPower//分类//bxtfidf"
outputkeyword =  os.path.join(outputfile,"kwds.txt");



train_data = []
train_target = []

test_data = []
test_target = []

dictlist = []
dict = {}
cnt = 0
labeldict = { u"T":1,u"F":0 }
random.seed(110)
with open(path,'rb' ) as f:
    for line in f:
        list = line.strip().split("\t")
        cnt += 1
        #if cnt > 30000 : break
        if len(list)>=2 :
            #print(list[1])
            rdm =  random.random()
            if list[0] not in dict:
                dict[list[0]] = len(dict)
            if rdm <= 0.75 :
                train_data.append(list[1])
                train_target.append(labeldict.get(list[0]))
                #train_target.append(int(list[0]))
            else:
                test_data.append(list[1])
                test_target.append(labeldict.get(list[0]))


                #test_target.append(int(list[0]))

print (cnt)
categories = dict.keys()
#print(train_data)
# split a training set and a test set
y_train, y_test = train_target , test_target


#停用词去除


print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(train_data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(train_data)
duration = time() - t0
#print("done in %fs at %0.3fMB/s" % (duration,  duration))
#print("n_samples: %d, n_features: %d" % X_train.shape)
#print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(test_data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()
#print(vectorizer.get_feature_names())
# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()
opts.select_chi2 = 3000
#opts.select_chi2 = None

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()
    cPickle.dump(ch2, open(os.path.join(outputfile, "ch2.data"), "wb"))
cPickle.dump((X_train,y_train),open(os.path.join(outputfile,"train.data"),"wb"))
cPickle.dump((X_test,y_test),open(os.path.join(outputfile,"test.data"),"wb"))
cPickle.dump(vectorizer,open(os.path.join(outputfile,"vectorizer.data"),"wb"))

if feature_names:
    feature_names = np.asarray(feature_names)
#print (",".join(feature_names))
ot = open(outputkeyword,"wb")
ot.writelines(",".join(feature_names))
ot.close()
#print(X_train)
def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


#print X_train
###############################################################################
# Benchmark classifiers
def benchmark(clf,name):
    print('_' * 80)
    print(time())
    print(tm.asctime( tm.localtime(tm.time()) ))
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    cPickle.dump(clf,open(os.path.join(outputfile,str(name)+".model"),"wb"))
    t0 = time()
    pred = clf.predict(X_test)

    predit = clf.predict(X_train)
   # clf.fit(X_train, y_train)
    test_time = time() - t0
    #print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    scoreit =  metrics.accuracy_score(y_train, predit)
    print("test set  accuracy:   %0.3f" % score)
    print("train set accuracy:   %0.3f" % scoreit)

    precision = metrics.precision_score(y_test,pred)
    precisionit = metrics.precision_score(y_train, predit)

    print("test set  precision: %0.3f" % precision)
    print("train set precision: %0.3f" % precisionit)


    recall = metrics.recall_score(y_test,pred)
    recallit = metrics.recall_score(y_train, predit)

    print("test set  recall: %0.3f" % recall)
    print("train set  recall: %0.3f" % recallit)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, category in enumerate(categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s"
                      % (category, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))


    #print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred,labels = [1,0]))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
# for clf, name in (
#         #(RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
#         #(Perceptron(n_iter=50), "Perceptron"),
#         #(PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
#         #(KNeighborsClassifier(n_neighbors=10), "kNN"),
#         (RandomForestClassifier(n_estimators=100), "Random forest")):
#     print('=' * 80)
#     print(name)
#     results.append(benchmark(clf,"Random_forest"))
print('=' * 80)
print("Perceptron")
results.append(benchmark(Perceptron(n_iter=150,n_jobs=-1),"Perceptron"))
results.append(benchmark(RandomForestClassifier(n_estimators=400), "Random forest"))
#results.append(benchmark(RandomForestClassifier(n_estimators=100),"RandomForest"))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3),"LinearSVC"+str(penalty)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=100,n_jobs=-1,
                                           penalty=penalty),"SGD_"+str(penalty)))

# Train SGD with Elastic Net penalty
# print('=' * 80)
# print("Elastic-Net penalty")
# results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                        penalty="elasticnet")))

# Train NearestCentroid without threshold
# print('=' * 80)
# print("NearestCentroid (aka Rocchio classifier)")
#results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01),"multinomialnb"))
results.append(benchmark(BernoulliNB(alpha=.01),"bernoulinb"))

# print('=' * 80)
# print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
# results.append(benchmark(Pipeline([
#   ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
#   ('classification', LinearSVC())
# ])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
plt.savefig(imagepath, format='png')