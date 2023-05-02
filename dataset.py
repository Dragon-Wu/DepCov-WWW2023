import os
import sys
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch
# import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

sys.path.append("..")
from utils import preprocess
from utils.tool_simple import (
    json_to_dict,
    dict_to_json,
    get_dict_part,
    seed_everything,
    init_logger,
    get_size_mem,
)
from sklearn.model_selection import train_test_split

class Normalize(object):
    def normalize(self, X_train, X_val, max_len):
        self.scaler = MinMaxScaler()
        len_train = X_train.shape[0]
        len_norm = len_train // 2
        X_train, X_val = X_train.reshape(len_train, -1), X_val.reshape(
            X_val.shape[0], -1
        )
        X_train[:len_norm,] = self.scaler.fit_transform(
            X_train[
                :len_norm,
            ]
        )
        X_train[len_norm:,] = self.scaler.transform(
            X_train[
                len_norm:,
            ]
        )
        X_val = self.scaler.transform(X_val)

        X_train, X_val = X_train.reshape(X_train.shape[0], max_len, -1), X_val.reshape(
            X_val.shape[0], max_len, -1
        )

        return (X_train, X_val)

    def inverse(self, X_train, X_val):
        X_train = self.scaler.inverse_transform(X_train)
        X_val = self.scaler.inverse_transform(X_val)

        return (X_train, X_val)

def process_data(dict_negative, dict_positive, max_length_tweet=21, tweet_mode='date'):
    list_data_negative, list_data_positive = [], []
    for username, data in dict_negative.items():
        if tweet_mode=='date':
            data = data["tweet_before_covid_date"] + data["tweet_after_covid_date"]
        else:
            data = data["tweet_before_covid"] + data["tweet_after_covid"]
        # get the true length of data
        len_real = len(data) if len(data)<max_length_tweet else max_length_tweet
        if max_length_tweet=='max':
            pass
        elif len(data) < max_length_tweet:
            data = data + ['']*(max_length_tweet-len(data))
        else:
            data = data[-max_length_tweet:]
        list_data_negative.append([data, len_real])
    num_negative = len(list_data_negative)

    for username, data in dict_positive.items():
        if tweet_mode=='date':
            data = data["tweet_before_covid_date"] + data["tweet_covid_depression_date"]
        else:
            data = data["tweet_before_covid"] + data["tweet_covid_depression"]
        # get the true length of data
        len_real = len(data) if len(data)<max_length_tweet else max_length_tweet
        if max_length_tweet=='max':
            pass
        elif len(data) < max_length_tweet:
            data = data + ['']*(max_length_tweet-len(data))
        else:
            data = data[-max_length_tweet:]
        list_data_positive.append([data, len_real])
    num_positive = len(list_data_positive)
    return num_negative, num_positive, list_data_negative, list_data_positive

def process_data_merge(dict_negative, dict_positive, max_length_tweet=21, tweet_mode='date'):
    list_data_negative, list_data_positive = [], []
    for username, data in dict_negative.items():
        if tweet_mode=='date':
            data = data["tweet_before_covid_date"] + data["tweet_after_covid_date"]
        else:
            data = data["tweet_before_covid"] + data["tweet_after_covid"]
        if max_length_tweet!='max' and len(data) > max_length_tweet:
            data = data[-max_length_tweet:]
        data = '.'.join(data) 
        list_data_negative.append(data)
    num_negative = len(list_data_negative)

    for username, data in dict_positive.items():
        if tweet_mode=='date':
            data = data["tweet_before_covid_date"] + data["tweet_covid_depression_date"]
        else:
            data = data["tweet_before_covid"] + data["tweet_covid_depression"]
        if max_length_tweet!='max' and len(data) > max_length_tweet:
            data = data[-max_length_tweet:]
        data = '.'.join(data) 
        list_data_positive.append(data)
    num_positive = len(list_data_positive)
    return num_negative, num_positive, list_data_negative, list_data_positive

class Dataset_covid_twitter(Dataset):
    def __init__(self, list_data, list_label, tokenizer, max_length_token):
        self.data = list_data
        self.label = list_label
        self.tokenizer = tokenizer
        self.max_length_token = max_length_token
        assert len(self.data) == len(self.label)

    def __getitem__(self, index):
        text_user = self.data[index][0]
        len_real = self.data[index][1]
        inputs = self.tokenizer(
            text_user,
            padding="max_length",
            truncation=True,
            max_length=self.max_length_token,
            return_tensors="pt",
        )
        return (
            inputs["input_ids"],
            inputs["attention_mask"],
            self.label[index],
            len_real,
        )

    def __len__(self):
        return len(self.data)


def get_dataloader(args, logger, valid=None):
    path_dir_data = args.path_dir_data
    dict_negative_all = json_to_dict(
        os.path.join(path_dir_data, f"dict_user_negative.json")
        # "/home/wjg/status_COVID19/data/dict_user_data_negative_pre.json"
        
    )
    dict_positive_all = json_to_dict(
        os.path.join(path_dir_data, f"dict_user_positive.json")
        # "/home/wjg/status_COVID19/data/dict_user_data_positive_pre.json"
    )
    logger.info(f"Loading negative: {len(dict_negative_all)}")
    logger.info(f"Loading positive: {len(dict_positive_all)}")

    args.num_case = int(args.num_case) if args.num_case!='max' else len(dict_positive_all) 

    dict_negative = get_dict_part(
        dict_negative_all,
        size = args.num_case * len(dict_negative_all) / len(dict_positive_all),
        shuffle=False,
    )
    dict_positive = get_dict_part(dict_positive_all, args.num_case, shuffle=False)

    num_negative, num_positive, list_data_negative, list_data_positive = process_data(
        dict_negative, dict_positive, 
        max_length_tweet=args.max_length_tweet, 
        tweet_mode=args.tweet_mode
    )

    logger.info(f"Each user have {list_data_negative[0][1]} content by mode {args.tweet_mode}")
    logger.info(f"Content like: ")
    logger.info(list_data_negative[0][0][0])
    logger.info(list_data_negative[0][0][-1])
    list_data = list_data_negative + list_data_positive
    list_label = [0] * num_negative + [1] * num_positive

    if not valid:
        # train : test = 8 : 2
        data_train, data_valid, label_train, label_valid = train_test_split(
            list_data,
            list_label,
            test_size=0.2,
            stratify=list_label,
            random_state=args.seed,
        )
        logger.info(
            f"Training Size: {len(data_train)}, {sum(label_train)} positive and {len(label_train)-sum(label_train)} negative"
        )
        logger.info(
            f"Testing Size: {len(data_valid)}, {sum(label_valid)} positive and {len(label_valid)-sum(label_valid)} negative"
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_student)
        dataset_train = Dataset_covid_twitter(
            list_data=data_train,
            list_label=label_train,
            tokenizer=tokenizer,
            max_length_token=args.max_length_token,
        )
        dataloader_train = DataLoader(dataset_train, batch_size=args.batchsize_train, drop_last=True)
        # , collate_fn=tokenizer_function
        dataset_valid = Dataset_covid_twitter(
            list_data=data_valid,
            list_label=label_valid,
            tokenizer=tokenizer,
            max_length_token=args.max_length_token,
        )
        dataloader_valid = DataLoader(dataset_valid, batch_size=args.batchsize_valid, drop_last=True)
        return dataloader_train, dataloader_valid

    else:
        # train : valid : test = 7 : 1 : 2
        data_train, data_valid_test, label_train, label_valid_test = train_test_split(
            list_data,
            list_label,
            test_size=0.3,
            stratify=list_label,
            random_state=args.seed,
        )
        data_valid, data_test, label_valid, label_test = train_test_split(
            data_valid_test,
            label_valid_test,
            test_size=0.66,
            stratify=label_valid_test,
            random_state=args.seed,
        )
        logger.info(
            f"Training Size: {len(data_train)}, {sum(label_train)} positive and {len(label_train)-sum(label_train)} negative"
        )
        logger.info(
            f"Evaluation Size: {len(data_valid)}, {sum(label_valid)} positive and {len(label_valid)-sum(label_valid)} negative"
        )
        logger.info(
            f"Testing Size: {len(data_test)}, {sum(label_test)} positive and {len(label_test)-sum(label_test)} negative"
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_student)
        dataset_train = Dataset_covid_twitter(
            list_data=data_train, list_label=label_train, tokenizer=tokenizer,
            max_length_token=args.max_length_token,
        )
        dataloader_train = DataLoader(dataset_train, batch_size=args.batchsize_train, drop_last=True)
        # , collate_fn=tokenizer_function

        dataset_valid = Dataset_covid_twitter(
            list_data=data_valid, list_label=label_valid, tokenizer=tokenizer,
            max_length_token=args.max_length_token,
        )
        dataloader_valid = DataLoader(dataset_valid, batch_size=args.batchsize_valid, drop_last=True)
        # , collate_fn=tokenizer_function
        dataset_test = Dataset_covid_twitter(
            list_data=data_test, list_label=label_test, tokenizer=tokenizer,
            max_length_token=args.max_length_token,
        )
        dataloader_test = DataLoader(dataset_test, batch_size=args.batchsize_valid, drop_last=True)
        # , collate_fn=tokenizer_function

        return dataloader_train, dataloader_valid, dataloader_test

def emb_padding_len(list_user_emb, len_emb_max=500, dim_emb=768):
    list_len_real = [0]*len(list_user_emb)
    for index in range(len(list_user_emb)):
        data = list_user_emb[index]
        len_emb = len(data)
        if len_emb <= len_emb_max:
            data_padding = np.concatenate((data, np.zeros((len_emb_max-len_emb, dim_emb), dtype=np.float32)))
        else:
            data_padding = data[:len_emb_max, :]
        len_real = len_emb if len_emb <= len_emb_max else len_emb_max
        list_user_emb[index] = [data_padding, len_real]
    return list_user_emb

class Dataset_emb(Dataset):
    def __init__(self, list_data, list_label):
        self.data = list_data
        self.label = list_label
        assert len(self.data) == len(self.label)

    def __getitem__(self, index):
        user_emb = self.data[index][0]
        len_real = self.data[index][1]
        label = self.label[index]
        return user_emb, label, len_real
        
    def __len__(self):
        return len(self.data)

def get_dataloader_emb(args, logger, valid=None):
    # direct
    path_dir_data = args.path_dir_data

    # loading
    with open(os.path.join(path_dir_data, f"emb_neg_{args.model_encoder}.p"), "rb") as fp:
        dict_emb_neg_all = pickle.load(fp)
    with open(os.path.join(path_dir_data, f"emb_pos_{args.model_encoder}.p"), "rb") as fp:
        dict_emb_pos_all = pickle.load(fp)
    
    args.num_case = int(args.num_case) if args.num_case!='max' else len(dict_emb_pos_all) 

    list_user_emb_neg = [
        dict_emb_neg_all[i]
        for i in list(dict_emb_neg_all.keys())[: args.num_case * 5]
    ]
    list_user_emb_pos = [
        dict_emb_pos_all[i] for i in list(dict_emb_pos_all.keys())[:args.num_case]
    ]

    logger.info(f"Loading negative: {len(list_user_emb_neg)}")
    logger.info(f"Loading positive: {len(list_user_emb_pos)}")

    dim_emb_sentence = list_user_emb_neg[0].shape[-1]

    # Padding item in list_user_emb_pos: [data_padding, len_real]
    list_user_emb_neg = emb_padding_len(list_user_emb_neg, len_emb_max=args.max_length_tweet, dim_emb=dim_emb_sentence)
    list_user_emb_pos = emb_padding_len(list_user_emb_pos, len_emb_max=args.max_length_tweet, dim_emb=dim_emb_sentence)
    # data and label and len
    list_data = list_user_emb_neg + list_user_emb_pos
    list_label = [0] * len(list_user_emb_neg) + [1] * len(list_user_emb_pos)
    
    logger.info(f"Each user have {args.max_length_tweet} sentence emb with dim {dim_emb_sentence}")

    if not valid:
        # train : test = 8 : 2
        data_train, data_valid, label_train, label_valid = train_test_split(
            list_data,
            list_label,
            test_size=0.2,
            stratify=list_label,
            random_state=args.seed,
        )
        logger.info(
            f"Training Size: {len(data_train)}, {sum(label_train)} positive and {len(label_train)-sum(label_train)} negative"
        )
        logger.info(
            f"Testing Size: {len(data_valid)}, {sum(label_valid)} positive and {len(label_valid)-sum(label_valid)} negative"
        )

        dataset_train = Dataset_emb(
            list_data=data_train,
            list_label=label_train,
        )
        dataloader_train = DataLoader(dataset_train, batch_size=args.batchsize_train, drop_last=True)
        # , collate_fn=tokenizer_function
        dataset_valid = Dataset_emb(
            list_data=data_valid,
            list_label=label_valid,
        )
        dataloader_valid = DataLoader(dataset_valid, batch_size=args.batchsize_valid, drop_last=True)

        return dataloader_train, dataloader_valid

    else:
        # train : valid : test = 7 : 1 : 2
        data_train, data_valid_test, label_train, label_valid_test = train_test_split(
            list_data,
            list_label,
            test_size=0.3,
            stratify=list_label,
            random_state=args.seed,
        )
        data_valid, data_test, label_valid, label_test = train_test_split(
            data_valid_test,
            label_valid_test,
            test_size=0.66,
            stratify=label_valid_test,
            random_state=args.seed,
        )
        logger.info(
            f"Training Size: {len(data_train)}, {sum(label_train)} positive and {len(label_train)-sum(label_train)} negative"
        )
        logger.info(
            f"Evaluation Size: {len(data_valid)}, {sum(label_valid)} positive and {len(label_valid)-sum(label_valid)} negative"
        )
        logger.info(
            f"Testing Size: {len(data_test)}, {sum(label_test)} positive and {len(label_test)-sum(label_test)} negative"
        )

        # train
        dataset_train = Dataset_emb(list_data=data_train, list_label=label_train)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batchsize_train, drop_last=True)    
        # valid
        dataset_valid = Dataset_emb(list_data=data_valid, list_label=label_valid)
        dataloader_valid = DataLoader(dataset_valid, batch_size=args.batchsize_valid, drop_last=True)
        # test
        dataset_test = Dataset_emb(list_data=data_test, list_label=label_test)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batchsize_valid, drop_last=True)

        return dataloader_train, dataloader_valid, dataloader_test