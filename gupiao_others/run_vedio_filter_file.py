# -*- coding:UTF-8 -*-
from __future__ import print_function

def gupiao_others():
    inputpath =u"D://workspace//python//classify_WeChat//data//financeoutput1_final//股票.txt"
    outputpath =u"D://workspace//python//classify_WeChat//data//financeoutput1_final//gupiao_others.txt"

    label1 = "香港市场"
    label2 = "国内市场"
    label3 = "美国市场"


    outfile = open(outputpath,"wb")
    with open(inputpath, "rb") as f:
        for line in f:
            splits = line.strip().split("\t")
            tag = splits[1]

            if tag.find(label1) < 0:
                if tag.find(label2) < 0:
                    if tag.find(label3) < 0:
                        outfile.writelines(line.strip() + "\t")
                        outfile.writelines("1\n")

            # if tag.find(label1) < 0:
            #     print(tag)
            #     train = []
            #
            #     outfile.writelines(line.strip()+"\t")
            #     outfile.writelines("1\n")
            #outfile.writelines(line.strip()+"\t"+lb.decode("utf-8").encode("utf-8")+"\n")
    outfile.close()

if __name__ == "__main__":
    gupiao_others()