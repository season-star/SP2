import ast
import json
from utils.util_data.util_general import *

import json
import re
import nltk
from fuzzywuzzy import fuzz
from collections import Counter
from copy import deepcopy


import re
punctuation = '!,;:?"\''
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation), '', text)
    return text.strip()

def handle_kb2triple(kb_arr,column_names,intent):
    kb_triple = []

    for single_kb in kb_arr:
        # if(intent[0]=="navigate"):
        subject = single_kb[0] #代表poi event location
        for object_index in range(len(single_kb)):
            tmp_triple=[]
            if(single_kb[object_index]==subject):continue
            tmp_triple.append(subject)
            tmp_triple.append(column_names[object_index]) #predicate
            tmp_triple.append(single_kb[object_index]) #value

            kb_triple.append(tmp_triple)
            # single_kb_triple[-1]=tmp_triple

        # elif(intent[0]=="schedule"):
        #     subject = single_kb[0]  # 代表event
        #     for object_index in range(len(single_kb)):
        #         tmp_triple = []
        #         if (single_kb[object_index] == subject): continue
        #         tmp_triple.append(subject)
        #         tmp_triple.append(column_names[object_index])  # predicate
        #         tmp_triple.append(single_kb[object_index])  # value
        #
        #         kb_triple.append(tmp_triple)
        #         # single_kb_triple[-1]=tmp_triple
        # elif(intent[0]=="weather"):
        #     pass

    return kb_triple

# file_train = data/Exk/train.json
def load_json_file(path,dataname):
    with open(path)as f:
        json_data = json.load(f)

    print(("Reading lines from {} and the length is {}".format(path,len(json_data))))
    data_detail = []

    for dialogue_id in range(len(json_data)):
        dialogue_data_detail = []

        driver_utt = ""
        driver_utt_slot = {}
        curr_intent =json_data[dialogue_id]['scenario']['task']['intent']

        # kb是适用于dialogue中的每一个turn的 在turn外围处理好就行
        kb=json_data[dialogue_id]['scenario']['kb']
        kb_items = kb['items']
        column_names = kb['column_names']
        curr_kb = []

        if(kb_items!=None):
            for item in kb_items:
                single_curr_kb = []
                for name in column_names:
                    single_curr_kb.append(item[name])
                curr_kb.append([])
                curr_kb[-1] =single_curr_kb

        # tmp_history=[]
        # print("==========================")
        # print(dialogue_id)

        dialogue_history = []
        turn_data_detail = {} #注意 是在每次dialogue_data_detail添加完turn之后,才清理的turn.在下方for循环的最后清理
        for turn_index,turn in enumerate(json_data[dialogue_id]['dialogue']):

            # 只有在assistant那个turn 才有slot标注 才开始处理
            slot_dict = {}
            turn_text_arr, turn_slot_arr, turn_intent_arr, turn_kb_arr, turn_cn_arr, turn_triple_arr = [], [], [], [], [], []


            curr_subject = turn['turn']
            dialogue_history.append(turn['data']['utterance']) #无论是driver还是assistant,都需要作为history加入

            # 我只需要对应 先是每个dial 然后是每个turn 就行
            if curr_subject == 'driver':
                driver_utt = removePunctuation(turn['data']['utterance']).split(" ")
                tmp_data_detail = {
                    'dial_id': str(dialogue_id),
                    'turn_id': str(turn_index),
                    'history': dialogue_history
                }
                # print(dialogue_id)
                # print(turn_index)
                # print(dialogue_history)
                # dialouge有问题
                turn_data_detail.update(tmp_data_detail)

            elif curr_subject == 'assistant':
                driver_utt_slot = turn['data']['slots']
                ass_utt = turn['data']['utterance']

                # 字典翻转
                re_driver_utt_slot = dict(zip(driver_utt_slot.values(),driver_utt_slot.keys()))
                # print(re_driver_utt_slot)
                # 'nearest': 'distance', 'parking garage': 'poi_type'
                for word in driver_utt:
                    if word != "":      slot_dict[word] = "O"

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
                # 原本的
                for key in slot_dict.keys():
                    turn_text_arr.append(key)

                for word_index,key in enumerate(slot_dict.keys()):
                    tmp_triple = [key,str(turn_index),str(word_index)]
                    turn_triple_arr.append(tmp_triple)

                for value in slot_dict.values():
                    turn_slot_arr.append(value)
                turn_intent_arr.append(curr_intent)
                turn_kb_arr=curr_kb
                turn_cn_arr=column_names

                tmp_data_detail = {
                    'slot': turn_slot_arr,
                    'text': turn_text_arr,
                    'intent': turn_intent_arr,
                    'kb': turn_kb_arr,
                    'cn': turn_cn_arr,
                    'triple': turn_triple_arr
                }
                turn_data_detail.update(tmp_data_detail)

                # kb_triple = handle_kb2triple(curr_kb, column_names, curr_intent)

                # print(turn_slot_arr)
                # for key,value in slot_dict.items():
                #     print("%s %s\n"%(key,value))
                # fw.write(curr_dia_intent+"\n\n")
            def add_single(result,single):
                result.append([])
                result[-1] = deepcopy(single)

            if(len(turn_slot_arr)!=0): #加判定是为了只有在turn为assistant时，才会有slot信息，才有字典，才能append得到的arr
                # 原本的写法 为了简便 引入addsingle函数
                # dialogue_data_detail.append([])
                # dialogue_data_detail[-1] = turn_data_detail
                add_single(dialogue_data_detail, turn_data_detail)
                turn_data_detail ={} #清理干净turn data detail 为了下一个user-ass做准备

        # 原本的写法
        # data_detail.append([])
        # data_detail[-1] = dialogue_data_detail

        add_single(data_detail,dialogue_data_detail)

        # 返回的是针对单句的slot的list
    # print(text_arr)

    # slot_arr的格式： 分成多个dialogue，每个dialogue中有多个turn，每个turn代表了一个list，含有slot值
    """
    [ [['O', 'O', 'B-distance', 'B-poi_type', 'I-poi_type'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-traffic_info', 'O', 'I-traffic_info', 'I-traffic_info', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O']]
    [['O', 'O', 'O', 'B-distance', 'B-poi_type', 'I-poi_type'], ['O', 'O', 'O', 'O'], ['O']]
    [['O', 'O', 'O', 'B-distance', 'O', 'O', 'O', 'O', 'O', 'B-poi_type', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-traffic_info', 'O', 'I-traffic_info', 'I-traffic_info'], ['O', 'O']]
    [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-event', 'O', 'O', 'B-date', 'O', 'B-time', 'O', 'O', 'O']]
    [['O', 'O', 'O', 'B-event', 'O', 'O', 'O', 'O'], ['O', 'O', 'O']]
    [['O', 'B-event', 'I-event'], ['O', 'O', 'O', 'B-date', 'O', 'B-time', 'I-time']]
    [['O', 'O', 'O', 'B-location', 'I-location'], ['O', 'O', 'O', 'B-weather_attribute', 'O', 'O'], ['O', 'O']]
    [['O', 'O', 'O', 'B-weather_attribute', 'O', 'B-location', 'O', 'O', 'B-date', 'I-date', 'I-date'], ['O', 'O', 'O']] ]
    """
    return data_detail
    # return text_arr,slot_arr,intent_arr,kb_arr,cn_arr,triple_arr, data_detail
    # print(slot_intent_arr)


def read_langs(file_name, max_line=None,file_type='txt'):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr = [], [], [], []
    max_resp_len = 0

    with open('data/KVR/kvret_entities.json') as f:
        global_entity = json.load(f)

    with open(file_name+'.txt') as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if '#' in line:
                    line = line.replace("#", "")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid))
                    context_arr += gen_u
                    conv_arr += gen_u

                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                    if task_type == "weather":
                        ent_idx_wet = gold_ent
                    elif task_type == "schedule":
                        ent_idx_cal = gold_ent
                    elif task_type == "navigate":
                        ent_idx_nav = gold_ent
                    ent_index = list(set(ent_idx_cal + ent_idx_nav + ent_idx_wet))

                    # Get local pointer position for each word in system response
                    ptr_index = []
                    for key in r.split():
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                        if (index):
                            index = max(index)
                        else:
                            index = len(context_arr)
                        ptr_index.append(index)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in
                                      context_arr] + [1]

                    sketch_response = generate_template(global_entity, r, gold_ent, kb_arr, task_type)

                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r,
                        'sketch_response': sketch_response,
                        'ptr_index': ptr_index + [len(context_arr)],
                        'selector_index': selector_index,
                        'ent_index': ent_index,
                        'ent_idx_cal': list(set(ent_idx_cal)),
                        'ent_idx_nav': list(set(ent_idx_nav)),
                        'ent_idx_wet': list(set(ent_idx_wet)),
                        'conv_arr': list(conv_arr),
                        'kb_arr': list(kb_arr),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type}
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    r = line
                    kb_info = generate_memory(r, "", str(nid))
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
            # 下一个dialogue
            else:
                cnt_lin += 1 #cnt_lin代表着dialogue的数量
                context_arr, conv_arr, kb_arr = [], [], []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_template(global_entity, sentence, sent_ent, kb_arr, domain):
    """
    Based on the system response and the provided entity table, the output is the sketch response.
    """
    sketch_response = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                if domain != 'weather':
                    for kb_item in kb_arr:
                        if word == kb_item[0]:
                            ent_type = kb_item[1]
                            break
                if ent_type == None:
                    for key in global_entity.keys():
                        if key != 'poi':
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                                ent_type = key
                                break
                        else:
                            poi_list = [d['poi'].lower() for d in global_entity['poi']]
                            if word in poi_list or word.replace('_', ' ') in poi_list:
                                ent_type = key
                                break
                sketch_response.append('@' + ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx)] + ["PAD"] * (MEM_TOKEN_SIZE - 4)
            sent_new.append(temp)
    else:
        sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def prepare_data_seq(batch_size=100):

    file_train = 'data/KVR/train'
    file_dev = 'data/KVR/dev'
    file_test = 'data/KVR/test'

    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)

    # load json file to get slot_arr
    path_json_train = file_train+'.json'
    path_json_dev=file_dev+'.json'
    path_json_test=file_test+'.json'

    text_arr,slot_arr = load_json_file(path_json_train)

        # print(pair_train[dialogue_index]['conv_arr'])
        # print(slot_arr[dialogue_index])

    # show_count = 0
    # for dialogue in pair_train:
    #     if(show_count>0):break
    #     show_count+=1
    #     for key,value in dialogue.items():
    #         print(key)
    #         print(value)
    #     print("\n")

    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)
    dev = get_seq(pair_dev, lang, batch_size, False)
    test = get_seq(pair_test, lang, batch_size, False)

    # print("Read %s sentence pairs train" % len(pair_train))
    # print("Read %s sentence pairs dev" % len(pair_dev))
    # print("Read %s sentence pairs test" % len(pair_test))
    # print("Vocab_size: %s " % lang.n_words)
    # print("Max. length of system response: %s " % max_resp_len)
    # print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, [], lang, max_resp_len
    # return pair_train,pair_dev,pair_test

def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    # print(pair)
    d = get_seq(pair, lang, batch_size, False)
    return d