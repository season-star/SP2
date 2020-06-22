"""
@Author		:           Lee, Qin
@StartTime	:           2018/08/13
@Filename	:           loader.py
@Software	:           Pycharm
@Framework  :           Pytorch
@LastModify	:           2019/05/07
"""

import os
import numpy as np
from copy import deepcopy
from collections import Counter
from collections import OrderedDict
from ordered_set import OrderedSet

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.util_data.util_exk import load_json_file
from utils.util_data.util_exk import handle_kb2triple


def digit_2_vis_text(index2ins, digit_text):
    vis_text = ""
    for digit in digit_text:
        vis_text += index2ins[digit] + " "
    return vis_text


class Alphabet(object):
    """
    Storage and serialization a set of elements.
    """

    def __init__(self, name, if_use_pad, if_use_unk):

        self.__name = name
        self.__if_use_pad = if_use_pad
        self.__if_use_unk = if_use_unk

        self.__index2instance = OrderedSet()
        self.__instance2index = OrderedDict()

        # Counter Object record the frequency
        # of element occurs in raw text.
        self.__counter = Counter()

        if if_use_pad:
            self.__sign_pad = "<PAD>"
            self.add_instance(self.__sign_pad)
        if if_use_unk:
            self.__sign_unk = "<UNK>"
            self.add_instance(self.__sign_unk)

    @property
    def name(self):
        return self.__name

    # @property
    def getindex2instance(self):
        return self.__index2instance

    # @property
    def getinstance2index(self):
        return self.__instance2index

    def add_instance(self, instance):
        """ Add instances to alphabet.

        1, We support any iterative data structure which
        contains elements of str type.

        2, We will count added instances that will influence
        the serialization of unknown instance.

        :param instance: is given instance or a list of it.
        """

        if isinstance(instance, (list, tuple)):
            for element in instance:
                self.add_instance(element)
            # for dial in instance:
            #     for turn in dial:
            #         self.add_instance(turn)
            return

        # We only support elements of str type.
        assert isinstance(instance, str)

        # count the frequency of instances.
        self.__counter[instance] += 1

        if instance not in self.__index2instance:
            self.__instance2index[instance] = len(self.__index2instance)
            self.__index2instance.append(instance)

    def get_index(self, instance):
        """ Serialize given instance and return.

        For unknown words, the return index of alphabet
        depends on variable self.__use_unk:

            1, If True, then return the index of "<UNK>";
            2, If False, then return the index of the
            element that hold max frequency in training data.

        :param instance: is given instance or a list of it.
        :return: is the serialization of query instance.
        """

        if isinstance(instance, (list, tuple)):
            return [self.get_index(elem) for elem in instance]
        assert isinstance(instance, str)

        try:
            return self.__instance2index[instance]
        except KeyError:
            if self.__if_use_unk:
                return self.__instance2index[self.__sign_unk]
            else:
                max_freq_item = self.__counter.most_common(1)[0][0]
                return self.__instance2index[max_freq_item]

    def get_instance(self, index):
        """ Get corresponding instance of query index.

        if index is invalid, then throws exception.

        :param index: is query index, possibly iterable.
        :return: is corresponding instance.
        """

        if isinstance(index, list):
            return [self.get_instance(elem) for elem in index]

        return self.__index2instance[index]

    def save_content(self, dir_path):
        """ Save the content of alphabet to files.

        There are two kinds of saved files:
            1, The first is a list file, elements are
            sorted by the frequency of occurrence.

            2, The second is a dictionary file, elements
            are sorted by it serialized index.

        :param dir_path: is the directory path to save object.
        """

        # Check if dir_path exists.
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        list_path = os.path.join(dir_path, self.__name + "_list.txt")
        with open(list_path, 'w') as fw:
            for element, frequency in self.__counter.most_common():
                fw.write(element + '\t' + str(frequency) + '\n')

        dict_path = os.path.join(dir_path, self.__name + "_dict.txt")
        with open(dict_path, 'w') as fw:
            for index, element in enumerate(self.__index2instance):
                fw.write(element + '\t' + str(index) + '\n')

    def __len__(self):
        return len(self.__index2instance)

    def __str__(self):
        return 'Alphabet {} contains about {} words: \n\t{}'.format(self.name, len(self), self.__index2instance)


class TorchDataset(Dataset):
    """
    Helper class implementing torch.utils.data.Dataset to
    instantiate DataLoader which deliveries data batch.
    """

    def __init__(self, dialogue_detail):
        # self.__text = text
        # self.__slot = slot
        # self.__intent = intent
        # self.__kb = kb
        # self.__text_triple = text_triple
        # self.__dial_id = dial_id
        # self.__turn_id = turn_id
        # self.__history = history
        self.__dialogue_detail = dialogue_detail

    def __getitem__(self, dialogue_index):
        # print(self.__dialogue_detail[dialogue_index])
        return self.__dialogue_detail[dialogue_index]  # 这样就实现了以dialogue为基本单元取batch
        # return self.__text[index], self.__slot[index], self.__intent[index],self.__kb[index], self.__text_triple[index], self.__dial_id[index], self.__turn_id[index], self.__history[index]

    def __len__(self):
        # Pre-check to avoid bug.
        # assert len(self.__text) == len(self.__slot)
        # assert len(self.__text) == len(self.__intent)
        # return len(self.__text)
        return len(self.__dialogue_detail)


class DatasetManager(object):

    def __init__(self, args):

        # Instantiate alphabet objects.
        self.__word_alphabet = Alphabet('word', if_use_pad=True, if_use_unk=True)
        self.__slot_alphabet = Alphabet('slot', if_use_pad=False, if_use_unk=False)
        self.__intent_alphabet = Alphabet('intent', if_use_pad=False, if_use_unk=False)
        self.__kb_alphabet = Alphabet('kb', if_use_pad=True, if_use_unk=False)
        self.__text_triple_alphabet = Alphabet('text_triple', if_use_pad=True, if_use_unk=False)

        self.__dial_id_alphabet = Alphabet('dial_id', if_use_pad=False, if_use_unk=False)
        self.__turn_id_alphabet = Alphabet('turn_id', if_use_pad=False, if_use_unk=False)
        self.__history_alphabet = Alphabet('history', if_use_pad=False, if_use_unk=False)

        # Record the raw text of dataset.
        # self.__text_word_data = {}
        # self.__text_slot_data = {}
        # self.__text_intent_data = {}
        # self.__text_kb_data = {}
        # self.__text_text_triple_data = {}
        # self.__text_dial_id_data = {}
        # self.__text_turn_id_data = {}
        # self.__text_history_data = {}

        self.__text_data_detail = {}

        # Record the serialization of dataset.
        # self.__digit_word_data = {}
        # self.__digit_slot_data = {}
        # self.__digit_intent_data = {}
        # self.__digit_kb_data = {}
        # self.__digit_text_triple_data = {}
        # self.__digit_dial_id_data = {}
        # self.__digit_turn_id_data = {}
        # self.__digit_history_data = {}

        self.__digit_data_detail = {}

        # 因为word slot的 alpha都一样，就先用word
        self.__index2instance = self.__kb_alphabet.getindex2instance()
        self.__instance2index = self.__kb_alphabet.getinstance2index()

        self.__args = args

        self.SINGLE_KB_SIZE = 3

    @property
    def show_text_data(self):
        return self.__text_slot_data

    @property
    def test_sentence(self):
        return deepcopy(self.__text_data_detail['test'])

    @property
    def word_alphabet(self):
        return deepcopy(self.__word_alphabet)

    @property
    def slot_alphabet(self):
        return deepcopy(self.__slot_alphabet)

    @property
    def intent_alphabet(self):
        return deepcopy(self.__intent_alphabet)

    @property
    def kb_alphabet(self):
        return deepcopy(self.__kb_alphabet)

    @property
    def text_triple_alphabet(self):
        return deepcopy(self.__text_triple_alphabet)

    @property
    def history_alphabet(self):
        return deepcopy(self.__history_alphabet)

    @property
    def num_epoch(self):
        return self.__args.num_epoch

    @property
    def batch_size(self):
        return self.__args.batch_size

    @property
    def learning_rate(self):
        return self.__args.learning_rate

    @property
    def l2_penalty(self):
        return self.__args.l2_penalty

    @property
    def save_dir(self):
        return self.__args.save_dir

    @property
    def intent_forcing_rate(self):
        return self.__args.intent_forcing_rate

    @property
    def slot_forcing_rate(self):
        return self.__args.slot_forcing_rate

    def get_alpha(self):
        return deepcopy(self.__kb_alphabet)

    def get_index2instance(self):
        return deepcopy(self.__instance2index)

    def get_mem_sentence_size(self):
        return deepcopy(self.__mem_sentence_size)

    def show_summary(self):
        """
        :return: show summary of dataset, training parameters.
        """

        print("Training parameters are listed as follows:\n")

        print('\tnumber of train sample:                    {};'.format(len(self.__text_word_data['train'])))
        print('\tnumber of dev sample:                      {};'.format(len(self.__text_word_data['dev'])))
        print('\tnumber of test sample:                     {};'.format(len(self.__text_word_data['test'])))
        print('\tnumber of epoch:						    {};'.format(self.num_epoch))
        print('\tbatch size:							    {};'.format(self.batch_size))
        print('\tlearning rate:							    {};'.format(self.learning_rate))
        print('\trandom seed:							    {};'.format(self.__args.random_state))
        print('\trate of l2 penalty:					    {};'.format(self.l2_penalty))
        print('\trate of dropout in network:                {};'.format(self.__args.dropout_rate))
        print('\tteacher forcing rate(slot)		    		{};'.format(self.slot_forcing_rate))
        print('\tteacher forcing rate(intent):		    	{};'.format(self.intent_forcing_rate))

        print("\nEnd of parameters show. Save dir: {}.\n\n".format(self.save_dir))

    # key function
    def quick_build(self):
        """
        Convenient function to instantiate a dataset object.
        """

        train_path = os.path.join(self.__args.data_dir, 'train.json')
        dev_path = os.path.join(self.__args.data_dir, 'dev.json')
        test_path = os.path.join(self.__args.data_dir, 'test.json')

        self.add_file(train_path, 'train', if_train_file=True)
        self.add_file(dev_path, 'dev', if_train_file=False)
        self.add_file(test_path, 'test', if_train_file=False)

        # Check if save path exists.
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        alphabet_dir = os.path.join(self.__args.save_dir, "alphabet")
        # print("alpha dir is %s"%(alphabet_dir)) #save/alphabet
        self.__word_alphabet.save_content(alphabet_dir)
        self.__slot_alphabet.save_content(alphabet_dir)
        self.__intent_alphabet.save_content(alphabet_dir)
        self.__kb_alphabet.save_content(alphabet_dir)

    # def get_dataset(self, data_name, is_digital):
    #     """ Get dataset of given unique name
    #
    #     :param data_name: is name of stored dataset.
    #     :param is_digital: make sure if want serialized data.
    #     :return: the required dataset.
    #     """
    #
    #     if is_digital:
    #         return self.__digit_word_data[data_name], \
    #                self.__digit_slot_data[data_name], \
    #                self.__digit_intent_data[data_name],\
    #                self.__digit_kb_data[data_name], \
    #                self.__digit_dial_id_data[data_name], \
    #                self.__digit_turn_id_data[data_name], \
    #                self.__digit_history_data[data_name]
    #
    #     else:
    #         return self.__text_word_data[data_name], \
    #                self.__text_slot_data[data_name], \
    #                self.__text_intent_data[data_name],\
    #                self.__text_kb_data[data_name], \
    #                self.__text_dial_id_data[data_name], \
    #                self.__text_turn_id_data[data_name], \
    #                self.__text_history_data[data_name]

    def add_file(self, file_path, data_name, if_train_file):
        # print(file_path)
        # 对他们都进行两个for循环的原因：原本的格式是按照dialogue进行分块的，但是因为这是单句的预测，所以不分dialogue了，都变成单句，为了之后还能用dialogue进行分块（加入local时），所以读取数据的时候，仍然多一个维度。
        # 为了现在单句的处理，所以需要将该维度降下来（我不太会unsqueze）所以两个for循环

        def reduce_dim(arr, info, type):
            # 只使用every dialogue的第一句
            # return only_use_first(arr,info)

            # global info
            for dial_info in arr:
                for turn_info in dial_info:
                    info.append(turn_info[type])
            return info

        def only_use_first(arr, info):
            for dial_info in arr:
                if (len(dial_info) != 0):
                    info.append(dial_info[0])
            return info

        def show_single(data_detail):
            show_count = 0
            for dial in data_detail:
                if (show_count > 0): break
                show_count += 1
                for turn in dial:
                    for key, value in turn.items():
                        print(key)
                        print(value)
                    print("-------------------")
                print("==========================\n")

        data_detail = load_json_file(file_path, data_name)
        """
        show data detail
        """
        # show_single(data_detail)
        # 到这之前 都是以dialogue作为基本单元的

        if if_train_file:
            for dial_data_detail in data_detail:
                for turn_data_detail in dial_data_detail:
                    # print(turn_data_detail)
                    # print(turn_data_detail['text'])
                    # print(turn_data_detail['uuid']) #排查有问题的dialogue
                    self.__word_alphabet.add_instance(turn_data_detail['text'])
                    self.__slot_alphabet.add_instance(turn_data_detail['slot'])
                    self.__intent_alphabet.add_instance(turn_data_detail['intent'])
                    self.__kb_alphabet.add_instance(turn_data_detail['kb'])  # 因为将五元组转成了三元组 所以不加column names
                    self.__text_triple_alphabet.add_instance(turn_data_detail['triple'])
                    self.__dial_id_alphabet.add_instance(
                        turn_data_detail['dial_id'])  # 为了使得接口统一,就还是将dial_id转化成str,然后添加进去.   其实不应该转成str,就应该是int,
                    self.__turn_id_alphabet.add_instance(turn_data_detail['turn_id'])
                    for single_history in turn_data_detail['history']:
                        self.__history_alphabet.add_instance(single_history)
                    # self.__history_alphabet = deepcopy(self.__word_alphabet)
        # Record the raw text of dataset.
        self.__text_data_detail[data_name] = deepcopy(data_detail)

        # Serialize raw text and stored it.
        # getindex不用弄 ,但是得思考,如何能够分别对text等进行index映射

        self.__digit_data_detail[data_name] = deepcopy(data_detail)  # 未经seriealed的data detail
        max_text_len = 0
        for digit_dialogue_index in range(len(self.__digit_data_detail[data_name])):
            for digit_turn_index in range(len(self.__digit_data_detail[data_name][digit_dialogue_index])):
                curr_detail = self.__digit_data_detail[data_name][digit_dialogue_index][
                    digit_turn_index]  # 当前dialogue中 当前turn的
                curr_text_len = len(curr_detail['text'])
                if curr_text_len > max_text_len:
                    max_text_len = curr_text_len

                self.__digit_data_detail[data_name][digit_dialogue_index][digit_turn_index][
                    'text'] = self.__word_alphabet.get_index(curr_detail['text'])
                if if_train_file:
                    self.__digit_data_detail[data_name][digit_dialogue_index][digit_turn_index][
                        'slot'] = self.__slot_alphabet.get_index(curr_detail['slot'])
                    self.__digit_data_detail[data_name][digit_dialogue_index][digit_turn_index][
                        'intent'] = self.__intent_alphabet.get_index(curr_detail['intent'])
                    self.__digit_data_detail[data_name][digit_dialogue_index][digit_turn_index][
                        'kb'] = self.__kb_alphabet.get_index(curr_detail['kb'])
                    self.__digit_data_detail[data_name][digit_dialogue_index][digit_turn_index][
                        'triple'] = self.__text_triple_alphabet.get_index(curr_detail['triple'])
                    self.__digit_data_detail[data_name][digit_dialogue_index][digit_turn_index][
                        'dial_id'] = self.__dial_id_alphabet.get_index(curr_detail['dial_id'])
                    self.__digit_data_detail[data_name][digit_dialogue_index][digit_turn_index][
                        'turn_id'] = self.__turn_id_alphabet.get_index(curr_detail['turn_id'])
                    self.__digit_data_detail[data_name][digit_dialogue_index][digit_turn_index][
                        'history'] = self.__history_alphabet.get_index(curr_detail['history'])
        self.__mem_sentence_size = max_text_len
        # print(self.__digit_data_detail[data_name][digit_dialogue_index][digit_turn_index]['text'])
        # for thing in self.__digit_data_detail[data_name]:
        #     print(thing)

    """
    @staticmethod
    def __read_file(file_path):
        #  Read data file of given path. param file_path: path of data file. return: list of sentence, list of slot and list of intent.
        texts, slots, intents = [], [], []
        text, slot = [], []

        with open(file_path, 'r') as fr:
            for line in fr.readlines():
                items = line.strip().split()

                if len(items) == 1: #到了intent那一行时，将之前所有积累的slot和intent append
                    texts.append(text)
                    slots.append(slot)
                    intents.append(items)

                    # clear buffer lists.
                    text, slot = [], []

                elif len(items) == 2:
                    text.append(items[0].strip())
                    slot.append(items[1].strip())

        return texts, slots, intents
    """

    def batch_delivery(self, data_name, batch_size=None, is_digital=True, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size

        if is_digital:
            dialogue_detail = self.__digit_data_detail[data_name]

        else:
            dialogue_detail = self.__text_data_detail[data_name]

        # print("==================batch delivery===============")
        # print(dialogue_detail[0])
        dataset = TorchDataset(dialogue_detail)

        return DataLoader(dataset, batch_size=batch_size, collate_fn=self.__collate_fn)

    def kb_padding(self, kb, pad_number=3):
        max_len_sinngle_kb = 0
        # pad = [0, 0, 0]
        pad = [0 for _ in range(pad_number)]
        padded_kb = []

        len_kb = [len(single_kb) for single_kb in kb]
        max_len_sinngle_kb = max(len_kb)

        for single_kb in kb:
            curr_kb = single_kb
            curr_len = len(single_kb)
            if (curr_len < max_len_sinngle_kb):
                for _ in range(max_len_sinngle_kb - curr_len):
                    curr_kb.append(pad)
            padded_kb.append([])
            padded_kb[-1] = curr_kb
        return padded_kb

    # @staticmethod
    def add_padding(self, texts, dial_id_batch=None, turn_id_batch=None,
                    history_batch=None, slot_batch=None, intent_batch=None, kb_batch=None, text_triple_batch=None,
                    digital=True):
        len_list = [len(text) for text in texts]
        max_len = max(len_list)

        # # Get sorted index of len_list.
        # sorted_index = np.argsort(len_list)[::-1]
        # trans_texts, seq_lens, trans_items = [], [], None
        # if items is not None:
        #     trans_items = [[] for _ in range(0, len(items))]
        #
        # # print(len_list)  #[5, 14, 6, 6, 4, 1, 11, 16, 2, 17, 8, 3, 3, 7]
        # # print(sorted_index) # [ 9  7  1  6 10 13  3  2  0  4 12 11  8  5]
        # print(dial_id_batch)
        # for (thing, type) in items:
        #     print(thing)
        # for index in sorted_index:
        #     seq_lens.append(deepcopy(len_list[index]))
        #     trans_texts.append(deepcopy(texts[index]))
        #     if digital:
        #         trans_texts[-1].extend([0] * (max_len - len_list[index]))
        #     else:
        #         trans_texts[-1].extend(['<PAD>'] * (max_len - len_list[index]))
        #
        #     # This required specific if padding after sorting.
        #     if items is not None:
        #         for item, (o_item, required) in zip(trans_items, items):
        #
        #             item.append(deepcopy(o_item[index]))
        #             if required:
        #                 if digital:
        #                     item[-1].extend([0] * (max_len - len_list[index]))
        #                 else:
        #                     item[-1].extend(['<PAD>'] * (max_len - len_list[index]))

        # handle kb 首先要kb和text对齐,其次才是padding

        sorted_texts = deepcopy(texts)
        sorted_dial_id = deepcopy(dial_id_batch)
        sorted_turn_id = deepcopy(turn_id_batch)
        sorted_history = deepcopy(history_batch)
        sorted_slot = deepcopy(slot_batch)
        sorted_intent = deepcopy(intent_batch)
        sorted_kb = deepcopy(kb_batch)
        sorted_text_triple = deepcopy(text_triple_batch)

        # print(texts)
        # print(len(texts))
        # for index in range(len(texts)):
        #     pass
        # for index in sorted_index:
        #     if dial_id_batch is not None:
        #         sorted_dial_id.append(dial_id_batch[index])
        #         sorted_turn_id.append(turn_id_batch[index])
        #         sorted_history.append(history_batch[index])
        #         sorted_kb.append(kb_batch[index])
        #         sorted_text_triple.append(text_triple_batch[index])
        #     # 匹配complete

        # padding
        kb_len_list = [len(single_kb) for single_kb in sorted_kb]  # print(kb_len_list)  #0, 40, 40, 40, 36, 0, 40, 40, 40, 40, 0, 36, 40, 40]
        max_kb_len = max(kb_len_list)

        triple_len_list = [len(single_triple) for single_triple in sorted_text_triple]
        max_triple_len = max(triple_len_list)
        digitalPAD = [0, 0, 0]
        textPAD = ['<PAD>', '<PAD>', '<PAD>']
        for index in range(len(sorted_texts)):
            #  dial_id和 turn_id不用填 就一行就行  主要是history和text_triple  . kb和text_triple固定三元组,history没定长, 需要两次填充
            if digital:
                sorted_texts[index].extend([0] * (max_len - len_list[index]))
                # slot 也不用 后面有expand list
                # intent也不用弄
                sorted_kb[index].extend([digitalPAD] * (max_kb_len - kb_len_list[index]))
                sorted_text_triple[index].extend([digitalPAD] * (max_triple_len - triple_len_list[index]))
            else:
                sorted_texts[index].extend(['<PAD>'] * (max_len - len_list[index]))

                sorted_kb[index].extend([textPAD] * (max_kb_len - kb_len_list[index]))
                sorted_text_triple[index].extend([textPAD] * (max_triple_len - triple_len_list[index]))

        # 单独弄history吧
        single_history_len_list = [[len(single_history) for single_history in history] for history in history_batch]
        # print(single_history_len_list)  #[[5, 16], [5, 16, 14, 25], [5, 16, 14, 25, 6, 3], [6, 7], [6, 7, 4, 19], [6, 7, 4, 19, 1, 1], [11, 11], [11, 11, 16, 14], [11, 11, 16, 14, 2, 6], [18, 5], [9, 7], [9, 7, 3, 2], [3, 5], [3, 5, 7, 14]]
        max_his_len = 0
        for single_his_len in single_history_len_list:
            his_len = max(single_his_len)
            if his_len > max_his_len:
                max_his_len = his_len
        # print(max_his_len)  #25

        row_his_len_list = [len(row_history) for row_history in history_batch]
        # print(row_his_len_list)
        max_row_his_len = max(row_his_len_list)
        # print(max_row_his_len)

        his_digit_PAD=[0]*max_his_len
        his_text_PAD = ['<PAD>']*max_his_len
        # print(his_PAD)
        for his_index in range(len(sorted_history)):
            for single_his_index in range(len(sorted_history[his_index])):
                curr_len = len(sorted_history[his_index][single_his_index])
                if digital:
                    sorted_history[his_index][single_his_index].extend([0] * (max_his_len - curr_len))
                else:
                    sorted_history[his_index][single_his_index].extend(['<PAD>'] * (max_his_len - curr_len))
            if digital:
                sorted_history[his_index].extend([his_digit_PAD]*(max_row_his_len-row_his_len_list[his_index]))
            else:
                sorted_history[his_index].extend([his_text_PAD]*(max_row_his_len-row_his_len_list[his_index]))


        # for i in range(len(sorted_dial_id)):
        #     print(sorted_dial_id[i])
        #     print(sorted_turn_id[i])
        #     print(trans_texts[i])
        #     # print(sorted_kb[i])
        #     print(sorted_history[i])
        #     print("===========")

        return sorted_texts, len_list, sorted_dial_id, sorted_turn_id, sorted_history, sorted_slot, sorted_intent, sorted_kb, sorted_text_triple
        # if items is not None:
        #     return trans_texts, trans_items, seq_lens, sorted_dial_id, sorted_turn_id, sorted_history, sorted_kb, sorted_text_triple
        # else:
        #     return trans_texts, seq_lens, sorted_dial_id, sorted_turn_id, sorted_history, sorted_kb, sorted_text_triple

    # helper function to instantiate a DataLoader Object.
    @staticmethod
    def __collate_fn(batch):
        # print("====================collatefn================")
        # print(batch)
        # print("-------")

        # info_type = 8 # 默认8
        # print(batch[0])
        info_type = len(batch[0][0])  # 如 slot text 这些有多少种类

        # for thing in batch:
        #     for t in thing:
        #         # print(t)
        #         print(len(t))


        modified_batch = [[] for _ in range(0, info_type)]
        for dialogue in batch:
            for turn in dialogue:
                # if(turn['uuid'] == "690bc2d7-f726-43db-a778-b4806a1ed743"):
                #     print(turn)
                curr_key_index = 0
                for key, value in turn.items():
                    modified_batch[curr_key_index].append(value)
                    # modified_batch[curr_key_index].append([value])
                    # print(value) #output uuid
                    curr_key_index += 1
                    # print(key)
                    # print(value)
                # print(turn)
            #     print("--------------")
            # print("==================")
        # for thing in modified_batch:
        #     print(thing)
        #     print("==================")

        return modified_batch
