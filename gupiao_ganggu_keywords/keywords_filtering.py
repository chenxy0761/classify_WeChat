# -*- coding:UTF-8 -*-

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
    #jieba_cut.load_userdict(u"股票关键词.txt")
    # = {}

    #outputfile = u"gptfidf"
    #output = open(u"股票_港股seg.txt",'wb')
    #input = u"股票_港股.txt"
    inputpath =u"../data/financeoutput1_final_08/financeoutput1_final_08.txt"
    outputpath =u"../data/financeoutput1_final_08/港股keywords.txt"
    #outputpath2 =u"F:\\work\\code\\classify-master_myself\\gupiao_ganggu_keywords\\classify\\港股去除文本.txt"
    
    kword_in = {}
    kword_not_in = {}
    keywordpath1 = u"港股关键词in.txt"
    keywordpath2 = u"港股关键词not_in.txt"
    
    with open(keywordpath1, "rb") as f:
        for line in f:
            word = line.strip()
            kword_in[word] = 0
            print word
    
    with open(keywordpath2, "rb") as f:
        for line in f:
            word = line.strip()
            kword_not_in[word] = 0
            print word
    
    outfile = open(outputpath,"wb")
#    outfile2 = open(outputpath2,"wb")
    
    def keywordMatch(kw,txt):
        count = 0.0
        for key in kw:
            text = txt
            offset = text.find(key)
            value = kw[key]
            while offset != -1:
                count = count + value
                text = text[offset + len(key):]
                offset = text.find(key)
                #print(key)
                #print (count)
            return count
    with open(inputpath, "rb") as f:
        linecount = 1
        for line in f:
            filterresult = False
            splits = line.strip().split("\t")
            count = 0
            count2 = 0
            countnumber = 0
            count = keywordMatch(kword_in,line)
            count2 = keywordMatch(kword_not_in,line)
            #print(linecount)
            countnumber = count - count2
            #print (countnumber) 
            if countnumber >= 4:
                filterresult = True
                #print("True")
                #else:
                    #print("False")
            if filterresult == True:
                outfile.writelines(line.strip()+"\n")
                print ("linecount:"+str(linecount))
            linecount += 1
    outfile.close()