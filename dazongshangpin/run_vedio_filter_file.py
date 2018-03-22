# -*- coding:UTF-8 -*-
from __future__ import print_function
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
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
import jieba
import random
import cPickle
import re

def dazongshangpin():
    outputfile = u"D://workspace//python//classify_WeChat//dazongshangpin//dzsptfidf"
    # outputfile = "dzsptfidf"
    X_train,y_train = cPickle.load(open(os.path.join(outputfile,"train.data"),"rb"))
    X_test,y_test = cPickle.load(open(os.path.join(outputfile,"test.data"),"rb"))
    vectorizer = cPickle.load(open(os.path.join(outputfile,"vectorizer.data"),"rb"))
    chi2  = cPickle.load(open(os.path.join(outputfile,"ch2.data"),"rb"))
    clf = cPickle.load(open(os.path.join(outputfile,"SGD_l2.model"),"rb"))
    #inputpath =u"E:\\项目需求\\JDPower\\分类\\4月份\\financeoutput1_final.txt"
    #outputpath =u"E:\\项目需求\\JDPower\\分类\\4月份\\大宗商品.txt"

    inputpath =u"D://workspace//python//classify_WeChat//data//financeoutput1_final//financeoutput1_final.txt"
    outputpath =u"D://workspace//python//classify_WeChat//data//financeoutput1_final//dazongshangpin.txt"
    # inputpath =u"..//data//financeoutput1_final//financeoutput1_final.txt"
    # outputpath =u"..//data//financeoutput1_final//大宗商品.txt"
    #outputpath2 =u"D:\\work\\code\\classify-master_myself\\dazongshangpin\\classify\\大宗商品去除文本_07.txt"

    label = "大宗商品"

    forbidkword = {}
    # load

    forbidpath = u"D://workspace//python//classify_WeChat//keyword.txt"
    # forbidpath = u"..//keyword.txt"
    with open(forbidpath, "rb") as f:
        for line in f:
            word = line.strip()
            forbidkword[word] = 0

    outfile = open(outputpath,"wb")
    with open(inputpath, "rb") as f:
        for line in f:
            splits = line.strip().split("\t")
            tag = splits[0]

            if tag.find(label) > -1 :
                print(tag)
                train = []
                #print (splits[-1])
                seg = jieba.cut(splits[-1], cut_all=False)
                #seglist = [i for i in seg]
                seglist = []
                for w in seg:
                    #print w
                    w = w.strip().encode("utf-8")
                    if w not in forbidkword:
                        if not re.match(r"\d+$", w):
                            seglist.append(w)
                train.append(" ".join(seglist))
                X_test = vectorizer.transform(train)
                X_test = chi2.transform(X_test)
                pred = clf.predict(X_test)
                #print(" ".join(pred))
                print (pred)
                lb = str(pred[0])
                #print(isinstance(lb, unicode))
                #print( lb.decode("gbk").encode("utf-8"))
                #outfile.writelines(lb+"\n")
                if lb == '1' :
                    outfile.writelines(line.strip()+"\t")
                    outfile.writelines(lb+"\n")
            #outfile.writelines(line.strip()+"\t"+lb.decode("utf-8").encode("utf-8")+"\n")
    outfile.close()

if __name__ == "__main__":
    dazongshangpin()