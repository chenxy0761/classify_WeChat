# -*- coding:UTF-8 -*-

import jieba_cut
import os
import re
import jieba_cut
import sys
import cPickle
import copy
import math
import pandas as pd
if __name__=="__main__":
    forbidkword = {}
    forbidpath = u"/Users/raymondmyz/Downloads/weixin/classify-master-88fa8bd7e293de195810725c2186554a7215fd51/keyword.txt"
    with open(forbidpath,"rb") as f :
        for line in f:
            word = line.strip()
            forbidkword[word] = 0


    #jieba_cut.load_userdict(kwodpath)
    stopdict = {}
    with open("/Users/raymondmyz/Downloads/weixin/classify-master-88fa8bd7e293de195810725c2186554a7215fd51/stopword.txt","rb") as f :
        for line in f :
            stopdict[line.strip()] = 1

    output = open(u"贵金属seg.txt",'wb')
    input = u"贵金属样本.txt"
    with open(input,'rb') as f:
        for line in f :
            #print line
            splits = line.strip().split("\t")
            #if len(splits)< 2 : continue
            lb = splits[0]#+","+splits[2]+","+splits[3]
            print lb
            content  = splits[1]
            jieba_seg = jieba_cut.cut(content, cut_all=False)
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