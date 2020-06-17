import sys

sys.path.append("/Users/season/Public/FYG/MyCode/SP")

import torch

import os
import json
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser()

#  python train.py -wed 256 -ehd 256 -aod 128

# Training parameters.
parser.add_argument('--data_dir', '-dd', type=str, default='data/KVR_demo/')

# if __name__=="__main__":
#     args = parser.parse_args()
#     path=args.data_dir
#
#     # ConvertKVR2ATIS(path,type="dev")
#
#     ConvertKB2txt(path)
if __name__=="__main__":
    data = open("train.txt","r",encoding="utf-8").readlines()
    p_data=[]
    count =0
    for d in data:
        if(d=="\n"):
            count +=1
    print("count is")
    print(count)