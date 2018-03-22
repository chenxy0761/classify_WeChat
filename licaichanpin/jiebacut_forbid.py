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


    forbidpath = u"..//keyword.txt"
    with open(forbidpath,"rb") as f :
        for line in f:
            word = line.strip()
            forbidkword[word] = 0
            #print word

    output = open(u"理财产品seg.txt",'wb')
    input = u"理财产品.txt"
    with open(input,'rb') as f:
        for line in f :
            #print line
            splits = line.strip().split("\t")
            #if len(splits)< 2 : continue
            lb = splits[0]#+","+splits[2]+","+splits[3]

            try:
                content  = splits[1]
            except Exception as err:
                print(lb + "\t"+content)

            if ((lb != "T") & (lb != "F")):
                continue
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