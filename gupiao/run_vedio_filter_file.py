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

def gupiao():
    #outputfile =  u"E:\\项目需求\\爬虫项目\\和硕爬虫\\tfidf\\"
    outputfile = u"gptfidf"
    outputfile = u"D://workspace//python//classify_WeChat//gupiao//gptfidf"
    # path = "/home/nlp/qdj/car/classify/tfidf/word_seg.txt"
    # imagepath = "/home/nlp/qdj/car/classify/tfidf/perf.png"
    # outputfile = "/home/nlp/qdj/car/classify/tfidf/"
    X_train,y_train = cPickle.load(open(os.path.join(outputfile,"train.data"),"rb"))
    X_test,y_test = cPickle.load(open(os.path.join(outputfile,"test.data"),"rb"))
    vectorizer = cPickle.load(open(os.path.join(outputfile,"vectorizer.data"),"rb"))
    chi2  = cPickle.load(open(os.path.join(outputfile,"ch2.data"),"rb"))
    clf = cPickle.load(open(os.path.join(outputfile,"SGD_l2.model"),"rb"))
    #inputpath =u"E:\\项目需求\\JDPower\\分类\\4月份\\financeoutput1_final.txt"
    #outputpath =u"E:\\项目需求\\JDPower\\分类\\4月份\\股票.txt"
    #F:\work\code\classify-master_myself\gupiao
    # inputpath =u"../data/financeoutput1_final/financeoutput1_final.txt"
    # outputpath =u"../data/financeoutput1_final/股票.txt"
    # outputpath2 =u"../data/financeoutput1_final_08/股票去除文本_08.txt"
    inputpath =u"D://workspace//python//classify_WeChat//data//financeoutput1_final//financeoutput1_final.txt"
    outputpath =u"D://workspace//python//classify_WeChat//data//financeoutput1_final//股票.txt"
    outputpath2 =u"D://workspace//python//classify_WeChat//data//financeoutput1_final//股票去除文本_08.txt"

    label = "股票"

    kword = {}
    keywordpath = u"D://workspace//python//classify_WeChat//gupiao//股票关键词.txt"
    # keywordpath = u"股票关键词.txt"
    #keywordpath_2=u"/Users/raymondmyz/Downloads/weixin/classify-master-88fa8bd7e293de195810725c2186554a7215fd51/gupiao/股票关键词.txt"
    jieba.load_userdict(keywordpath)

    outfile = open(outputpath,"wb")
    outfile2 = open(outputpath2,"wb")
    with open(inputpath, "rb") as f:
        for line in f:
            splits = line.strip().split("\t")
            tag_1 = splits[0]
            #print (tag_1)
            #tag_2 = splits[1]
            if tag_1 == "股票":
                train = []
                #print (splits[-1])
                seg = jieba.cut(splits[-1], cut_all=False)
                #seglist = [i for i in seg]
                seglist = []
                for w in seg:
                    #print w
                    w = w.strip().encode("utf-8")
                    seglist.append(w)
                #print  ( i ,",".join(seglist))
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
                    outfile.writelines(line.strip()+"\n")
                else:
                    outfile2.writelines(line.strip()+"\n")
                    #outfile.writelines(line.strip()+"\t"+lb.decode("utf-8").encode("utf-8")+"\n")
    outfile.close()

if __name__ == "__main__":
    gupiao()