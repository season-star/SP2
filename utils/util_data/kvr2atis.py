import sys
import json
import re
import nltk
from fuzzywuzzy import fuzz

sys.path.append("/Users/season/Public/FYG/MyCode/SP")
import utils.util_data.util_exk

def tokenize(sent):
    """
    Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    # for x in re.split("(\W+)?", sent):
    #     if x is not None:
    #         return x.strip()
    # return None
    return [x.strip() for x in re.split("(\W+)?", sent) if(x is not None and x.strip())]


import re
punctuation = '!,;:?"\''
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation), '', text)
    return text.strip()


# if __name__ == "__main__":
def ConvertKVR2ATIS(path,type="train"):
    # with open('data/KVR/kvret_entities.json') as f:
    #     global_entity = json.load(f)

    with open(path+"demo_"+type+".json")as f:
        train_data = json.load(f)


    fw= open(path+type+".txt","w")

    curr_dia=0
    for dialogue in range(len(train_data)):
        # print("-----------------------------")
        # fw.write("-----------------------------\n")
        # curr_dia+=1
        # if(curr_dia>3):
        #     break;

        driver_utt = ""
        driver_utt_slot = {}

        curr_dia_intent =train_data[dialogue]['scenario']['task']['intent']

        for turn in train_data[dialogue]['dialogue']:

            if (turn['turn'] == 'driver'):
                # driver_utt= turn['data']['utterance'].split(" ")
                driver_utt=removePunctuation(turn['data']['utterance']).split(" ")
            elif(turn['turn']=='assistant'):
                driver_utt_slot=turn['data']['slots']

            # 只有在assistant那个turn 才有slot标注 才开始处理
            if(turn['turn']=='assistant'):

                # 字典翻转
                re_driver_utt_slot = dict(zip(driver_utt_slot.values(),driver_utt_slot.keys()))
                # print(re_driver_utt_slot)
                # 'nearest': 'distance', 'parking garage': 'poi_type'

                slot_dict={}
                for word in driver_utt:
                    if (word != ""):
                        slot_dict[word]="O"

                # print(slot_dict)
                # {"where's": 'O', 'the': 'O', 'nearest': 'O', 'parking': 'O', 'garage': 'O'}

                findout = False
                for key,value in re_driver_utt_slot.items(): #key=parking garage    value =poi_type
                    begin =True
                    for word in key.split(" "): #word分别为parking 和 garage 第一遍
                        if(word in slot_dict.keys()):
                            findout=True
                            if(begin):
                                slot_dict[word]="B-"+value
                                begin=False
                            else:
                                slot_dict[word]="I-"+value
                        else:
                            break;
                    for word in key.split(" "):  # word分别为avoid heavy traffic 第一遍找不到
                        if(not findout):
                            for slot_dict_key in slot_dict.keys():
                                if(fuzz.ratio(word,slot_dict_key)>80):
                                    if(begin):
                                        slot_dict[slot_dict_key]="B-"+value
                                        begin=False
                                    else:
                                        slot_dict[slot_dict_key]="I-"+value

                for key,value in slot_dict.items():
                    fw.write("%s %s\n"%(key,value))
                    # print("%s %s"%(key,value))
                # print()
                fw.write(curr_dia_intent+"\n\n")

    print("over")
    f.close()

# 将json文件中的kb提出来
def ConvertKB2txt(path,type="train"):
    with open(path+"demo_"+type+".json")as f:
        data = json.load(f)
    for dialogue in data:
        # scenario=data[dialogue]['scenario']
        scenario = dialogue['scenario']

        intent = scenario['task']['intent']
        uuid=scenario['uuid']
        kb=scenario['kb']

        items=kb['items']
        column_names=kb['column_names']
        kb_title=kb['kb_title']

        print(kb_title)
