#!/bin/bash

#原本测试stackprop的命令
python -W ignore train.py -wed 256 -ehd 256 -aod 128

#测试读入exk
#python3 myTrain.py -lr=0.001 -l=1 -hdd=128 -dr=0.2 -dec=GLMP -bsz=8 -ds=kvr

#测试loadjson文件
#python utils/Test_module/test_util_exk.py
