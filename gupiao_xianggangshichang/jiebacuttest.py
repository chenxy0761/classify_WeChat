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
    # print("main")
    # stopwordfilename = u"G:\\sharefile\\stopword.txt"
    # cutwordfile = u"G:\\汽车样本与语料\\cutword.txt"
    # jieba_cut.load_userdict(cutwordfile)
    # info = "中型SUV,26.28  【自由光图片】jeep_自由光图片_汽车图片库_汽车之家 【自由光图片】jeep_自由光图片_汽车图片库_汽车之家"
    # jieba_seg = jieba_cut.cut(info , cut_all=False)
    # print ",".join(jieba_seg)
    # output = open(u"E://项目需求//自然语言处理--文本分类//母婴分类//xunlianyangben_seg.txt",'wb')
    # input = u"E://项目需求//自然语言处理--文本分类//母婴分类//xunlianyangben.txt"
    jieba.load_userdict(u"股票关键词.txt")
    stopdict = {}

    output = open(u"股票_港股seg.txt",'wb')
    input = u"股票_港股.txt"
    with open(input,'rb') as f:
        for line in f :
            #print line
            splits = line.strip().split("\t")
            #if len(splits)< 2 : continue
            lb = splits[0]#+","+splits[2]+","+splits[3]
            content  = splits[1]
            jieba_seg = jieba.cut(content, cut_all=False)
            outstr = " ".join(jieba_seg)
            strline = lb + "\t" + outstr.encode("utf-8")
            output.writelines(strline+"\n")
    output.close()