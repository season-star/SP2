"""
@Author		:           Lee, Qin
@StartTime	:           2018/08/13
@Filename	:           train.py
@Software	:           Pycharm
@Framework  :           Pytorch
@LastModify	:           2019/05/07
"""

from utils.module import ModelManager
from utils.loader import DatasetManager
from utils.process import Processor

import torch

import os
import json
import random
import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# Training parameters.
parser.add_argument('--data_dir', '-dd', type=str, default='data/KVR') #开发时用Exk 测试时用KVR
parser.add_argument('--save_dir', '-sd', type=str, default='save')
parser.add_argument("--random_state", '-rs', type=int, default=0)
parser.add_argument('--num_epoch', '-ne', type=int, default=200) #原本默认300
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.4)
parser.add_argument('--intent_forcing_rate', '-ifr', type=float, default=0.9)  #intent forcing rate
parser.add_argument("--differentiable", "-d", action="store_true", default=False)
parser.add_argument('--slot_forcing_rate', '-sfr', type=float, default=0.9) #slot forcing rate

if_exk = False
if  if_exk:
    parser.set_defaults(data_dir='data/Exk')
    parser.set_defaults(num_epoch=1)
    # parser.add_argument('--data_dir', '-dd', type=str, default='data/Exk')
    # parser.add_argument('--num_epoch', '-ne', type=int, default=1)

# My Parameters
parser.add_argument('--use_cuda','-uc',default=False)
parser.add_argument('--use_mem','-umn',help="if use memory network(global)",default=True) #是否使用mem
parser.add_argument('--ctrnn_embedding_dim','-ced',help="Context Rnn hidden size",default=256)
parser.add_argument('--mem_embedding_dim','-med',help="memory network dimension",default=256)
parser.add_argument('--max_hops','-mh',default=6)

# model parameters.
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=8)
parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=64)
parser.add_argument('--intent_decoder_hidden_dim', '-idhd', type=int, default=64)
parser.add_argument('--attention_hidden_dim', '-ahd', type=int, default=1024)
parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)

if __name__ == "__main__":
    args = parser.parse_args()

    # Save training and model parameters.
    if not os.path.exists(args.save_dir):
        os.system("mkdir -p " + args.save_dir)

    log_path = os.path.join(args.save_dir, "param.json") #parameter保存在这里\
    with open(log_path, "w") as fw:
        fw.write(json.dumps(args.__dict__, indent=True))

    # Fix the random seed of package random.
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # Fix the random seed of Pytorch when using GPU. GPU的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
        torch.cuda.manual_seed(args.random_state)

    # Fix the random seed of Pytorch when using CPU. CPU的随机种子
    torch.manual_seed(args.random_state)
    torch.random.manual_seed(args.random_state)

    # Instantiate a dataset object. 输入dataset
    dataset = DatasetManager(args)
    dataset.quick_build() #这里打印的三次

    mem_sentence_size = dataset.get_mem_sentence_size()

    # dataset.show_summary() #原本显示 暂时隐藏

    # Instantiate a network model object. 实例化model  输入构建好的dataset中的word slot的length
    model = ModelManager(args, len(dataset.word_alphabet),len(dataset.slot_alphabet),len(dataset.intent_alphabet),len(dataset.kb_alphabet),len(dataset.history_alphabet),mem_sentence_size=mem_sentence_size)
    # model.show_summary() #原本显示 暂时隐藏

    # To train and evaluate the models.  在这里真正进行dataset的输入
    process = Processor(dataset, model, args.batch_size)
    process.train()

    print("-------------------------VALIDATE----------------------------------")
    print('\nAccepted performance: ' +
        str(
            Processor.validate(  #slot_f1, intent_acc, sent_acc
                os.path.join(args.save_dir, "model/model.pkl"),
                os.path.join(args.save_dir, "model/dataset.pkl"),
                args.batch_size
            )
        ) + " at test dataset;\n")




    """
    ,
  {
    "dialogue": [
      {
        "turn": "driver",
        "data": {
          "end_dialogue": false,
          "utterance": "Car I wanna eat at some Chinese restaurant: please pick a route with least traffic, better if there will be no traffic at all"
        }
      },
      {
        "turn": "assistant",
        "data": {
          "end_dialogue": false,
          "requested": {
            "distance": false,
            "traffic_info": true,
            "poi_type": true,
            "address": false,
            "poi": false
          },
          "slots": {
            "traffic_info": "least traffic",
            "poi_type": "Chinese "
          },
          "utterance": "Tai Pan is 1 miles away."
        }
      },
      {
        "turn": "driver",
        "data": {
          "end_dialogue": false,
          "utterance": "What is the address?"
        }
      },
      {
        "turn": "assistant",
        "data": {
          "end_dialogue": false,
          "requested": {
            "distance": false,
            "traffic_info": false,
            "poi_type": false,
            "address": true,
            "poi": false
          },
          "slots": {},
          "utterance": "Tai Pan is located at 830 Almanor Lane"
        }
      },
      {
        "turn": "driver",
        "data": {
          "end_dialogue": false,
          "utterance": "Okay great. Could I have some details about the traffic there?"
        }
      },
      {
        "turn": "assistant",
        "data": {
          "end_dialogue": false,
          "requested": {
            "distance": false,
            "traffic_info": true,
            "poi_type": true,
            "address": false,
            "poi": true
          },
          "slots": {
            "traffic_info": "details about the traffic"
          },
          "utterance": "There is moderate traffic on our way but since it's just 1 miles away it shouldn't take too long to reach"
        }
      },
      {
        "turn": "driver",
        "data": {
          "end_dialogue": false,
          "utterance": "Sounds great, thank you. "
        }
      },
      {
        "turn": "assistant",
        "data": {
          "end_dialogue": true,
          "requested": {
            "distance": false,
            "traffic_info": false,
            "poi_type": false,
            "address": false,
            "poi": false
          },
          "slots": {},
          "utterance": "You're welcome!"
        }
      }
    ],
    "scenario": {
      "kb": {
        "items": [
          {
            "distance": "5 miles",
            "traffic_info": "heavy traffic",
            "poi_type": "parking garage",
            "address": "610 Amarillo Ave",
            "poi": "Stanford Oval Parking"
          },
          {
            "distance": "2 miles",
            "traffic_info": "heavy traffic",
            "poi_type": "home",
            "address": "10 ames street",
            "poi": "home"
          },
          {
            "distance": "6 miles",
            "traffic_info": "heavy traffic",
            "poi_type": "pizza restaurant",
            "address": "776 Arastradero Rd",
            "poi": "Dominos"
          },
          {
            "distance": "3 miles",
            "traffic_info": "no traffic",
            "poi_type": "rest stop",
            "address": "465 Arcadia Pl",
            "poi": "Four Seasons"
          },
          {
            "distance": "2 miles",
            "traffic_info": "heavy traffic",
            "poi_type": "coffee or tea place",
            "address": "145 Amherst St",
            "poi": "Teavana"
          },
          {
            "distance": "1 miles",
            "traffic_info": "moderate traffic",
            "poi_type": "chinese restaurant",
            "address": "830 Almanor Ln",
            "poi": "tai pan"
          },
          {
            "distance": "2 miles",
            "traffic_info": "moderate traffic",
            "poi_type": "friends house",
            "address": "580 Van Ness Ave",
            "poi": "toms house"
          }
        ],
        "column_names": [
          "poi",
          "poi_type",
          "address",
          "distance",
          "traffic_info"
        ],
        "kb_title": "location information"
      },
      "task": {
        "intent": "navigate"
      },
      "uuid": "a847b1e8-f8e1-4029-9fb9-b4e7f92c2401"
    }
  },
  {
    "dialogue": [
      {
        "turn": "driver",
        "data": {
          "end_dialogue": false,
          "utterance": "what will the weather be like tomorrow"
        }
      },
      {
        "turn": "assistant",
        "data": {
          "end_dialogue": false,
          "requested": {
            "weather_attribute": true,
            "date": true,
            "location": false
          },
          "slots": {
            "date": "tomorrow"
          },
          "utterance": "Where can I give you the weather forecast for?"
        }
      },
      {
        "turn": "driver",
        "data": {
          "end_dialogue": false,
          "utterance": "I want to know if it will be humid in Compton tomorrow"
        }
      },
      {
        "turn": "assistant",
        "data": {
          "end_dialogue": false,
          "requested": {
            "weather_attribute": true,
            "date": true,
            "location": true
          },
          "slots": {
            "weather_attribute": "humid",
            "date": "tomorrow",
            "location": "Compton"
          },
          "utterance": "The forecast does not state that it will be humid in Compton tomorrow."
        }
      },
      {
        "turn": "driver",
        "data": {
          "end_dialogue": false,
          "utterance": "Thanks."
        }
      },
      {
        "turn": "assistant",
        "data": {
          "end_dialogue": true,
          "requested": {
            "weather_attribute": false,
            "date": false,
            "location": false
          },
          "slots": {},
          "utterance": "No problem!"
        }
      }
    ],
    "scenario": {
      "kb": {
        "items": [
          {
            "monday": "snow, low of 40F, high of 50F",
            "tuesday": "frost, low of 20F, high of 40F",
            "friday": "rain, low of 40F, high of 50F",
            "wednesday": "clear skies, low of 60F, high of 70F",
            "thursday": "drizzle, low of 90F, high of 100F",
            "sunday": "clear skies, low of 90F, high of 100F",
            "location": "manhattan",
            "saturday": "rain, low of 70F, high of 80F",
            "today": "monday"
          },
          {
            "monday": "hot, low of 70F, high of 90F",
            "tuesday": "clear skies, low of 80F, high of 90F",
            "friday": "windy, low of 50F, high of 60F",
            "wednesday": "overcast, low of 80F, high of 100F",
            "thursday": "clear skies, low of 50F, high of 60F",
            "sunday": "clear skies, low of 50F, high of 70F",
            "location": "grand rapids",
            "saturday": "overcast, low of 70F, high of 80F",
            "today": "monday"
          },
          {
            "monday": "windy, low of 40F, high of 60F",
            "tuesday": "overcast, low of 60F, high of 80F",
            "friday": "snow, low of 40F, high of 60F",
            "wednesday": "cloudy, low of 60F, high of 80F",
            "thursday": "raining, low of 60F, high of 70F",
            "sunday": "cloudy, low of 40F, high of 50F",
            "location": "compton",
            "saturday": "overcast, low of 70F, high of 90F",
            "today": "monday"
          },
          {
            "monday": "rain, low of 70F, high of 90F",
            "tuesday": "raining, low of 20F, high of 30F",
            "friday": "snow, low of 40F, high of 50F",
            "wednesday": "windy, low of 60F, high of 70F",
            "thursday": "foggy, low of 40F, high of 60F",
            "sunday": "blizzard, low of 30F, high of 50F",
            "location": "san jose",
            "saturday": "overcast, low of 90F, high of 100F",
            "today": "monday"
          },
          {
            "monday": "rain, low of 40F, high of 60F",
            "tuesday": "cloudy, low of 80F, high of 90F",
            "friday": "blizzard, low of 20F, high of 40F",
            "wednesday": "drizzle, low of 30F, high of 40F",
            "thursday": "overcast, low of 70F, high of 80F",
            "sunday": "frost, low of 40F, high of 60F",
            "location": "oakland",
            "saturday": "overcast, low of 50F, high of 70F",
            "today": "monday"
          },
          {
            "monday": "raining, low of 50F, high of 60F",
            "tuesday": "dry, low of 70F, high of 90F",
            "friday": "rain, low of 80F, high of 90F",
            "wednesday": "clear skies, low of 60F, high of 80F",
            "thursday": "snow, low of 80F, high of 90F",
            "sunday": "clear skies, low of 50F, high of 60F",
            "location": "mountain view",
            "saturday": "rain, low of 80F, high of 100F",
            "today": "monday"
          },
          {
            "monday": "foggy, low of 90F, high of 100F",
            "tuesday": "raining, low of 20F, high of 30F",
            "friday": "windy, low of 60F, high of 80F",
            "wednesday": "overcast, low of 70F, high of 90F",
            "thursday": "raining, low of 80F, high of 90F",
            "sunday": "drizzle, low of 80F, high of 100F",
            "location": "cleveland",
            "saturday": "dry, low of 50F, high of 70F",
            "today": "monday"
          }
        ],
        "column_names": [
          "location",
          "monday",
          "tuesday",
          "wednesday",
          "thursday",
          "friday",
          "saturday",
          "sunday",
          "today"
        ],
        "kb_title": "weekly forecast"
      },
      "task": {
        "intent": "weather"
      },
      "uuid": "1e4d4726-d4c3-4118-84aa-e5caf6b561bb"
    }
  },
    """