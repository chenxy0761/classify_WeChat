#！-*- coding:utf-8 -*-
from baoxian.run_vedio_filter_file import baoxian
from cunkuan.run_vedio_filter_file import cunkuan
from daikejingwai.run_vedio_filter_file import daikejingwai
from dazongshangpin.run_vedio_filter_file import dazongshangpin
from guoneikaifangjijin.run_vedio_filter_file import guoneikaifangjijin
from gupiao.run_vedio_filter_file import gupiao
from gupiao_guoneishichang.run_vedio_filter_file import gupiao_guoneishichang
from gupiao_meiguoshichang.run_vedio_filter_file import gupiao_meiguoshichang
from gupiao_others.run_vedio_filter_file import gupiao_others
from gupiao_xianggangshichang.run_vedio_filter_file import gupiao_xianggangshichang
from haiwaifangchan.run_vedio_filter_file import haiwaifangchan
from licaichanpin.run_vedio_filter_file import licaichanpin
from waihui.run_vedio_filter_file import waihui
from zhaiquan.run_vedio_filter_file import zhaiquan
def run():
    baoxian()
    cunkuan()
    daikejingwai()
    dazongshangpin()
    guoneikaifangjijin()
    gupiao()
    gupiao_guoneishichang()
    gupiao_meiguoshichang()
    gupiao_xianggangshichang()
    gupiao_others()
    haiwaifangchan()
    licaichanpin()
    waihui()
    zhaiquan()

if __name__ == "__main__":
    run()