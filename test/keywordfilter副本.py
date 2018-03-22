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
import re
guoneishichangkeywordpath = u"国内市场keyword.txt"
meigukeywordpath = u"美股keyword.txt"
ganggukeywordpath = u"港股keyword.txt"
inputfilepath = u"/Users/raymondmyz/Downloads/weixin/07分类/股票07.txt"
guoneikeyword = []
meigukeyword = []
ganggukeyword = []
outputpath = u"/Users/raymondmyz/Downloads/weixin/07分类/国内市场test07.txt"
outfile = open(outputpath,"wb")
def tsplit(string, delimiters):
    """Behaves str.split but supports multiple delimiters."""

    delimiters = tuple(delimiters)
    stack = [string,]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i+j, _substring)

    return stack
with open(guoneishichangkeywordpath,"rb") as f:
	for line in f:
		item = line.strip()
		#print (item)
		guoneikeyword.append(item)
		#print(guoneikeyword)
with open(meigukeywordpath,"rb") as f:
	for line in f:
		item = line.strip()
		meigukeyword.append(item)
		#print (meigukeyword)
with open(ganggukeywordpath,"rb") as f:
	for line in f:
		item = line.strip()
		ganggukeyword.append(item)
		#print (ganggukeyword)
with open(inputfilepath,"rb") as f:
	linecount = 1 
	for line in f:
		item = line.strip().split("\t",3)
		label_1 = item[0]
		label_2 = item[1]
		label_3 = item[2]
		content = item[3]
		if (label_2 == '国内市场'):
			#print (content)
			print ('linecount'+'\t'+str(linecount))
			linecount += 1
			#print (line)
			
			sentence = []
			splitresult = tsplit(content,('。','？','；'))
			for i in range(len(splitresult)):
				sentence.append(splitresult[i])
			sentencedict = {}
			sentencetotal = float(len(splitresult))
			sentencedict['guoneisentence'] = 0.0
			sentencedict['meigusentence'] = 0.0
			sentencedict['ganggusentence'] = 0.0
			othersencente = 0.0
			for i in range(len(sentence)):
				#print (sentence[i])
				havelabeled = False
				if havelabeled == False:
					for j in range(len(guoneikeyword)):
						keyword = guoneikeyword[j]
						offset = sentence[i].find(keyword)
						if offset != -1:
							sentencedict['guoneisentence'] +=1.0
							havelabeled = True
							#print ('have found'+keyword+'\t'+'label'+'\t'+'国内sentence')
							break
				if havelabeled == False:
					for j in range(len(meigukeyword)):
						keyword = meigukeyword[j]
						offset = sentence[i].find(keyword)
						if offset != -1:
							sentencedict['meigusentence'] +=1.0
							havelabeled = True
							#print ('have found'+keyword+'\t'+'label'+'\t'+'美股sentence')
							break
				if havelabeled == False:
					for j in range(len(ganggukeyword)):
						keyword = ganggukeyword[j]
						offset = sentence[i].find(keyword)
						if offset != -1:
							sentencedict['ganggusentence'] +=1.0
							havelabeled = True
							#print ('have found'+keyword+'\t'+'label'+'\t'+'港股sentence')
							break
				if havelabeled == False:
					othersencente += 1.0
					#print ('no keyword found'+'label'+'\t'+'其他sentence')
			#print ('sentencetotal'+str(sentencetotal))
			#print ('guoneisentence'+str(sentencedict['guoneisentence']))
			#print ('meigusentence'+str(sentencedict['meigusentence']))
			#print ('ganggusentence'+str(sentencedict['ganggusentence']))
			#print ('othersencente'+str(othersencente))
			print (max(sentencedict.items(), key=lambda x: x[1]))
			predictlabel = max(sentencedict.items(), key=lambda x: x[1])[0]
			if (predictlabel == 'guoneisentence') and sentencedict['guoneisentence']/sentencetotal > 0.3 :
				outfile.writelines(line.strip()+"\n")



