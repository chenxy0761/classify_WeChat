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
import classifyconfig

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

            train_data.append(list[1])
            train_target.append(labeldict.get(list[0]))


print (cnt)
categories = dict.keys()
#print(train_data)
# split a training set and a test set
y_train = train_target


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

print("Extracting features from the test data using the same vectorizer")
t0 = time()

duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, duration))
print()

if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()
opts.select_chi2 = classifyconfig.chi2

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()
    cPickle.dump(ch2, open(os.path.join(outputfile, "ch2.data"), "wb"))
cPickle.dump((X_train,y_train),open(os.path.join(outputfile,"train.data"),"wb"))
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



results = []

print('=' * 80)
print("Perceptron")
results.append(benchmark(Perceptron(n_iter=150,n_jobs=-1),"Perceptron"))
results.append(benchmark(RandomForestClassifier(n_estimators=400), "Random forest"))
for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3),"LinearSVC"+str(penalty)))
    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=100,n_jobs=-1,
                                           penalty=penalty),"SGD_"+str(penalty)))

print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01),"multinomialnb"))
results.append(benchmark(BernoulliNB(alpha=.01),"bernoulinb"))
