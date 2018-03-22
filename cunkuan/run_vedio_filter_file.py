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
import jieba_cut
import random
import cPickle
def cunkuan():
    #inputpath =u"E://项目需求//JDPower//分类//4月份//financeoutput1_final.txt"
    #outputpath =u"E://项目需求//JDPower//分类//4月份//存款.txt"
    inputpath =u"D://workspace//python//classify_WeChat//data//financeoutput1_final//financeoutput1_final.txt"
    outputpath =u"D://workspace//python//classify_WeChat//data//financeoutput1_final//cunkuan.txt"

    label = "存款"


    outfile = open(outputpath,"wb")
    with open(inputpath, "rb") as f:
        for line in f:
            splits = line.strip().split("\t")
            tag = splits[0]

            if tag.find(label) > -1 :
                print(tag)
                train = []

                outfile.writelines(line.strip()+"\t")
                outfile.writelines("1\n")
            #outfile.writelines(line.strip()+"\t"+lb.decode("utf-8").encode("utf-8")+"\n")
    outfile.close()

if __name__ == "__main__":
    cunkuan()