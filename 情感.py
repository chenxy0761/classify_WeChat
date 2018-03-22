# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import re
import jieba
import codecs
from pyexcel_xlsx import get_data
from pyexcel_xlsx import save_data


################################step 0 ################################################################
#修改各词库的路径
#stopword_path = 'D:/ING/微博微信语义分析/stop words/stopwords.txt'
#degreeword_path = 'D:/ING/微博微信语义分析/emotion analysis/emotion/degreewords.txt'
#sentimentword_path = 'D:/ING/微博微信语义分析/emotion analysis/BosonNLP_sentiment_score/BosonNLP_sentiment_score_keep1.txt'
degreeword_path = u"D:\\下载软件\\QQ聊天记录\\degreewords.txt"
sentimentword_path=u"D:\\下载软件\\QQ聊天记录\\BosonNLP_sentiment_score_keep1.txt"
#加载新词库
jieba.load_userdict(u'D:\\下载软件\\QQ聊天记录\\stock_dict.txt')

# 停用词列表
#stopword_file = open(stopword_path,"r").readlines()
#stopwords = [word.replace("\n","") for word in stopword_file]

#否定词表
notword = [u'不',u'没',u'无',u'非',u'莫',u'弗',u'勿',u'毋',u'未',u'否',u'别',u'無',u'休',u'难道']

#程度词表
degreeword_file = open(degreeword_path).readlines()
degree_dict = {}
for word in degreeword_file:
    word = word.replace("\n","").split(" ")
    degree_dict[word[0]] = word[1]
    print(word[1])

#情感词表
"""
sentimentword_file = open(sentimentword_path, encoding='utf-8').readlines()
sentiment_dict = {}
for word in sentimentword_file:
    word = word.replace("\n","").split(" ")
    sentiment_dict[word[0]] = word[1]
"""
sentimentword_file = open(sentimentword_path).readlines()
sentiment_dict = {}
for word in sentimentword_file:
    word = word.replace("\n","").split(",")
    sentiment_dict[word[0]] = word[1]
print("Great!We have loaded all word lists!")

################################step 1 ################################################################
"""
step 1 : 分词且去除停用词
"""
def sent2wordloc(sentence):
    wordlist = []
    wordloc = {}
    #wordlist = [word for word in jieba_cut.cut(sentence) if word not in stopwords]
    wordlist = [word for word in jieba.cut(sentence)]
    wordloc = {word:loc for loc, word in enumerate(wordlist)}
    return wordlist,wordloc
print("The function sent2word is defined")
print("ok")
import pandas as pd
def read_xls_file(file, sheet_n):
    xls_data = get_data(file)[sheet_n]
    print("Get data type:(%s)" %(type(xls_data)))
    xls_df = pd.DataFrame(xls_data)
    return xls_df
################################step 2 ################################################################
"""
step 2 : 针对分词结果，定位情感词、否定词及程度词
"""
def wordclassify(sentence):
    wordlist, wordloc = sent2wordloc(sentence)
    sentimentloc, notloc, degreeloc, othersloc = {}, {}, {}, {}
    for word in wordloc.keys():
        if word in sentiment_dict.keys() and word not in notword and word not in degree_dict.keys():
            sentimentloc[wordloc[word]] = sentiment_dict[word]
        elif word in notword and word not in degree_dict.keys():
            notloc[wordloc[word]] = -1
        elif word in degree_dict.keys():
            degreeloc[wordloc[word]] = degree_dict[word]
        else:
            othersloc[wordloc[word]] = 0
    sentimentloc = sorted(sentimentloc.items(), key=lambda x:x[0])
    return sentimentloc, notloc, degreeloc, othersloc, wordlist, wordloc
print("The function wordclassify is defined")

################################step 3 ################################################################
"""
step 3 : 情感打分
"""
def sentscore(sentence):
    sentimentloc, notloc, degreeloc, othersloc, wordlist, wordloc = wordclassify(sentence)
    w = 1
    score = 0
    for i in range(len(sentimentloc)):
        wl = list(sentimentloc[i])[0]
        ww = list(sentimentloc[i])[1]
        score += w*float(ww)
        if i < (len(sentimentloc)-1):
            for j in range(wl+1,list(sentimentloc[i+1])[0]):
                if j in notloc.keys():
                    w *= -1
                elif j in degreeloc.keys():
                    w *= float(degreeloc[j])
    return score
print("The function sentscore is defined")


################################step 4 ################################################################

def read_xls_file(file, sheet_n):
    xls_data = get_data(file)[sheet_n]
    print("Get data type:(%s)" %(type(xls_data)))
    xls_df = pd.DataFrame(xls_data)
    return xls_df

#filepath = "D:/ING/微博微信语义分析/weibo/result_file_2.xlsx"
filepath= u"E://项目需求//JDPower//资产状态//Product.xlsx"
data = read_xls_file(filepath,"Sheet1")

#test
score = []
for i in range(len(data)):
   print(sentscore(data.iloc[i,4]),data.iloc[i,4])

#data['score'] = pd.Series(list(map(lambda x: sentscore(x), data.iloc[:,4])))
#print("ok")





#data.to_csv('E://result_file_2_withscore.csv', index=False, header=None)