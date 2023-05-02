#!/usr/bin/env bash
# experiment setting
# the model 
mode="StudentAttnTransPos_his_offline"
# Only content or mood:
#   SelfMLP / SelfAttn / SelfTrans / SelfTransPos / 
# Mood2Content:
#   StudentMLP_offline / StudentAttn_offline / StudentAttnTrans_offline / StudentAttnTransPos_offline 
# Mood and Content:
#   MoodAndContent
# HAN
#    GRUHANClassifier 

# anything your want describe
description=""

# dataset setting 
# path of data
path_dir_data=""
# how many case used to train 
num_case="max"
# tweet_mode: tweet(single tweet) or date(daily tweet)
tweet_mode="date" # date / tweet
# how many tweets/daily tweets used to predict (the invesre order) 
max_length_tweet=28
# the truncated token of tweets/daily tweets
max_length_token=256
# loss setting 
weight_distill=1
weight_clf=1
ratio_weight_dynamic=1
weight_min=0
# model setting
model_teacher=""
model_student=""
# the strategy of emb generation
pool_strategy="mean" # "cls" / "mean"
freeze_teacher=True
freeze_student=False

# encoder setting
model_encoder="CovBertLarge"
# Bert / BertSt / BertEmo
# CovBertLarge / CovBertLargeSt256 / CovBertLargeEmo256 / CovBertLargeTSA
dim_emb=1024
num_head_encoder=16
num_layer_encoder=12
hidden_dropout_prob=0.1
# learning
scheduler="get_cosine_schedule_with_warmup"
auto_lr_find=False
learning_rate=5e-5
weight_decay=0.01
adam_epsilon=1e-8
# epoch and batch
epochs=10
batchsize_train=2
batchsize_valid=256
step_accumulate=8
step_logging=1
step_eval=0.1
earlystop_patience=20
use_swa=False
# metric
metric='val_auroc'
# device
device="4657"
# seed
seed=42
echo "Start running"
# nohup command: run the job in the bg and never stop until killing or completion
nohup python main.py \
    --mode=${mode} \
    --description=${description} \
    --path_dir_data=${path_dir_data} \
    --num_case=${num_case} \
    --tweet_mode=${tweet_mode} \
    --max_length_tweet=${max_length_tweet} \
    --max_length_token=${max_length_token} \
    --weight_distill=${weight_distill} \
    --weight_clf=${weight_clf} \
    --ratio_weight_dynamic=${ratio_weight_dynamic} \
    --weight_distill_min=${weight_min} \
    --model_teacher=${model_teacher} \
    --model_student=${model_student} \
    --pool_strategy=${pool_strategy} \
    --freeze_teacher=${freeze_teacher} \
    --freeze_student=${freeze_student} \
    --model_encoder=${model_encoder} \
    --dim_emb=${dim_emb} \
    --num_head_encoder=${num_head_encoder} \
    --num_layer_encoder=${num_layer_encoder} \
    --epochs=${epochs} \
    --batchsize_train=${batchsize_train} \
    --batchsize_valid=${batchsize_valid} \
    --step_accumulate=${step_accumulate} \
    --step_logging=${step_logging} \
    --step_eval=${step_eval} \
    --earlystop_patience=${earlystop_patience} \
    --hidden_dropout_prob=${hidden_dropout_prob} \
    --learning_rate=${learning_rate} \
    --auto_lr_find=${auto_lr_find} \
    --weight_decay=${weight_decay} \
    --adam_epsilon=${adam_epsilon} \
    --use_swa=${use_swa} \
    --seed=${seed} \
    --device=${device} \
    1>out/${mode}_${tweet_mode}_${max_length_tweet}_${model_encoder}_${seed}_${num_head_encoder}_${num_layer_encoder}.out 2>&1 &