import sys
sys.path.append("/Users/season/Public/FYG/MyCode/SP")
from utils.util_data.util_exk import *
from tqdm import tqdm

if __name__ == "__main__":
    data_detail=load_json_file('data/Exk/train.json')
    print(data_detail[0])
    # for thing in data_detail[0]:
    #     print(thing)

    # print(text)
    # print(slot)
'''

# 分批次读取数据
    train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq( batch_size=4)
    count_print=0

    pbar = tqdm(enumerate(train), total=len(train))
    for i, data in pbar:
        pass
        # print(data)
        # model.train_batch(data, int(args['clip']), reset=(i == 0))
        # pbar.set_description("hello")

'''