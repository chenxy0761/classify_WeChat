# -*- coding:UTF-8 -*-

import jieba
import os
import re
import jieba
import sys
import cPickle
import copy
import math
import pandas as pd
if __name__=="__main__":

    forbidkword = {}
    #load

    #forbidpath = u"E://项目需求//JDPower//分类//keyword.txt"
    #这里的关键词，严格意义上是指噪音词汇，需要在分词的时候剔除
    forbidpath = u"..//keyword.txt"
    with open(forbidpath,"rb") as f :
        for line in f:
            word = line.strip()
            forbidkword[word] = 0
            print word


    #jieba_cut.load_userdict(kwodpath)
    stopdict = {}
    with open(u"..//stopword.txt","rb") as f :
        for line in f :
            stopdict[line.strip()] = 1

    #output = open(u"E://项目需求//JDPower//分类//保险seg.txt",'wb')
    output = open(u"保险seg.txt",'wb')

    #input = u"E://项目需求//JDPower//分类//保险.txt"
    input = u"保险.txt"
    with open(input,'rb') as f:
        for line in f :
            #print line
            splits = line.strip().split("\t")
            #if len(splits)< 2 : continue
            lb = splits[0]#+","+splits[2]+","+splits[3]
            content  = splits[1]
            jieba_seg = jieba.cut(content, cut_all=False)
            seg = []
            for w in jieba_seg:
                #print w
                w = w.strip().encode("utf-8")
                if w not in forbidkword:
                    if not re.match(r"\d+$",w):
                        seg.append(w)
                    #print w
            outstr = " ".join(seg)
            strline = lb + "\t" + outstr
            output.writelines(strline+"\n")
    output.close()