# import 
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (PackedSequence, pack_padded_sequence,
                                pad_packed_sequence)

# transformers
from transformers import *

# defined method
from model.func import *

class SelfMLP(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        # student model
        self.model_clf = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        # freeze encoder
        if args.freeze_student==True:
            for name, param in self.model_clf.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze student model !")
        # attn
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.model_clf.config.hidden_size, out_features=2)
        # some config
        self.logger.info(f"core model: SelfMLP")
        self.logger.info(f"hidden_size: {self.model_clf.config.hidden_size}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        
    def forward(self, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        for idx_user, [user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            emb_tweets_student = self.model_clf(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            # get the emb of tweet by pool strategy 
            emb_tweets_student = get_pool_emb(emb_tweets_student['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)
            # get the emb of user by mean pooling
            emb_tweets_student = torch.mean(emb_tweets_student, dim=0)
            batch_emb_user_student = emb_tweets_student if idx_user==0 else torch.vstack((batch_emb_user_student, emb_tweets_student))
        # dropout 
        batch_emb_user_student = self.dropout(batch_emb_user_student)
        batch_logits_user = self.classifier(batch_emb_user_student)
        return {"logits": batch_logits_user}

class SelfAttn(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        # student model
        self.model_clf = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        # freeze encoder
        if args.freeze_student==True:
            for name, param in self.model_clf.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze student model !")
        # attn
        self.attn_ff = nn.Linear(self.model_clf.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.model_clf.config.hidden_size, out_features=2)
        # some config
        self.logger.info(f"core model: SelfAttn")
        self.logger.info(f"hidden_size: {self.model_clf.config.hidden_size}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        
    def forward(self, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        for idx_user, [user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # print(f"len_real:{len_real} user_input_ids: {user_input_ids.shape}, user_attention_mask: {user_attention_mask.shape}")
            emb_tweets_student = self.model_clf(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            # emb layer
            emb_tweets_student = get_pool_emb(emb_tweets_student['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)
            # attention layer
            scores_attn = torch.softmax(self.attn_ff(emb_tweets_student).squeeze(-1), -1)
            # weighted sum [hidden_size, ]
            emb_user_student = scores_attn @ emb_tweets_student
            batch_emb_user_student = emb_user_student if idx_user==0 else torch.vstack((batch_emb_user_student, emb_user_student))
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout 
        batch_emb_user_student = self.dropout(batch_emb_user_student)
        batch_logits_user = self.classifier(batch_emb_user_student)
        return {"logits": batch_logits_user,'scores_attn': batch_scores_attn}

class SelfTrans(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        # student model
        self.tweet_encoder = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        # attn-transformer emb of text and emb of pos
        self.hidden_dim = args.dim_emb
        num_heads = args.num_head_encoder
        num_trans_layers = args.num_layer_encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: SelfTrans")
        self.logger.info(f"hidden_size: {self.tweet_encoder.config.hidden_size}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        self.logger.info(f"TransformerEncoder: {num_heads} num_heads {num_trans_layers} num_trans_layers")

    def forward(self, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        for idx_user, [user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # print(f"len_real:{len_real} user_input_ids: {user_input_ids.shape}, user_attention_mask: {user_attention_mask.shape}")
            # tweet emb
            emb_tweets_user = self.tweet_encoder(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            # user emb
            emb_user = get_pool_emb(emb_tweets_user['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)
            emb_user = self.user_encoder(emb_user).squeeze(1)
            # attention layer
            scores_attn = torch.softmax(self.attn_ff(emb_user).squeeze(-1), -1)
            emb_user = scores_attn @ emb_user
            batch_emb_user = emb_user if idx_user==0 else torch.vstack((batch_emb_user, emb_user))
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout 
        batch_emb_user = self.dropout(batch_emb_user)
        batch_logits_user = self.classifier(batch_emb_user)
        return {"logits": batch_logits_user, "scores_attn": batch_scores_attn}

class SelfTransPos(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        # student model
        self.tweet_encoder = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        # config 
        self.hidden_dim = args.dim_emb
        num_heads = args.num_head_encoder
        num_trans_layers = args.num_layer_encoder
        # emb of pos
        self.pos_emb = nn.Parameter(torch.Tensor(args.max_length_tweet, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        # attn-transformer emb of text 
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: SelfTransPos")
        self.logger.info(f"hidden_size: {self.tweet_encoder.config.hidden_size}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        self.logger.info(f"TransformerEncoder: {num_heads} num_heads {num_trans_layers} num_trans_layers")

    def forward(self, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        for idx_user, [user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # print(f"len_real:{len_real} user_input_ids: {user_input_ids.shape}, user_attention_mask: {user_attention_mask.shape}")
            # emb tweet
            emb_tweets_user = self.tweet_encoder(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            # emb user
            emb_user = get_pool_emb(emb_tweets_user['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)
            # attention layer
            emb_user =  emb_user + self.pos_emb[-len_real:, ]
            emb_user = self.user_encoder(emb_user).squeeze(1)
            scores_attn = torch.softmax(self.attn_ff(emb_user).squeeze(-1), -1)
            emb_user = scores_attn @ emb_user
            batch_emb_user = emb_user if idx_user==0 else torch.vstack((batch_emb_user, emb_user))
            scores_attn = scores_attn.detach().cpu()
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout 
        batch_emb_user = self.dropout(batch_emb_user)
        batch_logits_user = self.classifier(batch_emb_user)
        return {"logits": batch_logits_user, "scores_attn": batch_scores_attn}

class SelfTransPosMean(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        # student model
        self.tweet_encoder = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        # config 
        self.hidden_dim = args.dim_emb
        num_heads = args.num_head_encoder
        num_trans_layers = args.num_layer_encoder
        # attn-transformer emb of text 
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: SelfTransPos")
        self.logger.info(f"hidden_size: {self.tweet_encoder.config.hidden_size}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        self.logger.info(f"TransformerEncoder: {num_heads} num_heads {num_trans_layers} num_trans_layers")

    def forward(self, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        for idx_user, [user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # print(f"len_real:{len_real} user_input_ids: {user_input_ids.shape}, user_attention_mask: {user_attention_mask.shape}")
            # emb tweet
            emb_tweets_user = self.tweet_encoder(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            # emb user
            emb_user = get_pool_emb(emb_tweets_user['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)
            # attention layer
            emb_user = self.user_encoder(emb_user) # 28, 768
            emb_user = torch.mean(emb_user, axis=0)
            batch_emb_user = emb_user if idx_user==0 else torch.vstack((batch_emb_user, emb_user))
            scores_attn = torch.zeros(self.config.max_length_tweet).to(emb_user.device)
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout 
        batch_emb_user = self.dropout(batch_emb_user)
        batch_logits_user = self.classifier(batch_emb_user)
        return {"logits": batch_logits_user, "scores_attn": batch_scores_attn}

class SelfTransPosHis(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        # student model
        self.tweet_encoder = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        # config 
        self.hidden_dim = args.dim_emb
        num_heads = args.num_head_encoder
        num_trans_layers = args.num_layer_encoder
        # emb of pos
        self.pos_emb = nn.Parameter(torch.Tensor(args.max_length_tweet, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        # attn-transformer emb of text 
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim+self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: SelfTransPos")
        self.logger.info(f"hidden_size: {self.tweet_encoder.config.hidden_size}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        self.logger.info(f"TransformerEncoder: {num_heads} num_heads {num_trans_layers} num_trans_layers")

    def forward(self, batch_user_his, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        for idx_user, [user_his, user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_user_his, batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # print(f"len_real:{len_real} user_input_ids: {user_input_ids.shape}, user_attention_mask: {user_attention_mask.shape}")
            # emb tweet
            emb_tweets_user = self.tweet_encoder(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            # emb user
            emb_user = get_pool_emb(emb_tweets_user['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)
            # attention layer
            emb_user = torch.hstack(emb_user, user_his)
            # emb_user =  emb_user + self.pos_emb[-len_real:, ]
            emb_user = self.user_encoder(emb_user).squeeze(1)
            scores_attn = torch.softmax(self.attn_ff(emb_user).squeeze(-1), -1)
            emb_user = scores_attn @ emb_user
            batch_emb_user = emb_user if idx_user==0 else torch.vstack((batch_emb_user, emb_user))
            scores_attn = scores_attn.detach().cpu()
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout 
        batch_emb_user = self.dropout(batch_emb_user)
        batch_logits_user = self.classifier(batch_emb_user)
        return {"logits": batch_logits_user, "scores_attn": batch_scores_attn}

class StudentMLP(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        # teacher model 
        self.model_teacher = AutoModelForSequenceClassification.from_pretrained(self.config.model_teacher, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading teacher model: {self.config.model_teacher}")
        # student model
        self.model_student = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model: {self.config.model_student}")
        # freeze the encoder
        if args.freeze_teacher==True:
            for name, param in self.model_teacher.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze student model !")
        if args.freeze_student==True:
            for name, param in self.model_student.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze teacher model !")
        self.hidden_dim = self.model_student.config.hidden_size
        # clf
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: StudentMLP")
        self.logger.info(f"hidden_size: {self.hidden_dim}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        self.logger.info(f"classifier: mlp")
        
    def forward(self, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        # input:
        # batch_user_input_ids and batch_user_attention_mask: {batch, max_length_tweet, max_length_token}
        for idx_user, [user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # emb_tweets_student: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            emb_tweets_student = self.model_student(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            emb_tweets_student = get_pool_emb(emb_tweets_student['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)

            # emb_tweets_teacher: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            with torch.no_grad():
                self.model_teacher.eval()
                emb_tweets_teacher = self.model_teacher(
                    input_ids=user_input_ids, attention_mask=user_attention_mask, 
                )
                emb_tweets_teacher = get_pool_emb(emb_tweets_teacher['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)

            # batch_emb_tweets_student: {batch, max_length_tweet, dim_of_bert}
            batch_emb_tweets_student = emb_tweets_student if idx_user==0 else torch.cat((batch_emb_tweets_student, emb_tweets_student))
            batch_emb_tweets_teacher = emb_tweets_teacher if idx_user==0 else torch.cat((batch_emb_tweets_teacher, emb_tweets_teacher))

            # mean pooling of each tweet: emb_user_student: {dim_of_bert}
            emb_user_student = torch.mean(emb_tweets_student, dim=0)
            # emb_user_student = torch.mean(emb_tweets_student, dim=0)
            batch_emb_user_student = emb_user_student if idx_user==0 else torch.vstack((batch_emb_user_student, emb_user_student))
        # dropout 
        batch_emb_user_student = self.dropout(batch_emb_user_student)
        logits_user = self.classifier(batch_emb_user_student)

        return {
            "logits": logits_user,
            "emb_teacher": batch_emb_tweets_teacher, 
            "emb_student": batch_emb_tweets_student, 
        }


class StudentAttn(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        # teacher model
        self.model_teacher = AutoModelForSequenceClassification.from_pretrained(self.config.model_teacher, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading teacher model:{self.config.model_teacher}")
        # student model
        self.model_student = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        self.hidden_dim = self.model_student.config.hidden_size
        # freeze the encoder
        if args.freeze_teacher==True:
            for name, param in self.model_teacher.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze student model !")
        if args.freeze_student==True:
            for name, param in self.model_student.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze teacher model !")
        # attn
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: StudentAttn")
        self.logger.info(f"hidden_size: {self.hidden_dim}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        
    def forward(self, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        # input:
        # batch_user_input_ids and batch_user_attention_mask: {batch, max_length_tweet, max_length_token}
        for idx_user, [user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # emb_tweets_student: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            emb_tweets_student = self.model_student(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            emb_tweets_student = get_pool_emb(emb_tweets_student['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)

            # emb_tweets_teacher: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            with torch.no_grad():
                self.model_teacher.eval()
                emb_tweets_teacher = self.model_teacher(
                    input_ids=user_input_ids, attention_mask=user_attention_mask, 
                )
                emb_tweets_teacher = get_pool_emb(emb_tweets_teacher['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)

            # batch_emb_tweets_student: {batch, max_length_tweet, dim_of_bert}
            batch_emb_tweets_student = emb_tweets_student if idx_user==0 else torch.cat((batch_emb_tweets_student, emb_tweets_student))
            batch_emb_tweets_teacher = emb_tweets_teacher if idx_user==0 else torch.cat((batch_emb_tweets_teacher, emb_tweets_teacher))

            # attention layer
            scores_attn = torch.softmax(self.attn_ff(emb_tweets_student).squeeze(-1), -1)
            # weighted sum [hidden_size, ]
            emb_user_student = scores_attn @ emb_tweets_student
            batch_emb_user_student = emb_user_student if idx_user==0 else torch.vstack((batch_emb_user_student, emb_user_student))
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout and clf
        batch_emb_user_student = self.dropout(batch_emb_user_student)
        logits_user = self.classifier(batch_emb_user_student)
        return {
            "logits": logits_user,
            "emb_teacher": batch_emb_tweets_teacher, 
            "emb_student": batch_emb_tweets_student, 
            "scores_attn": batch_scores_attn
        }


class StudentAttnTrans(nn.Module):
    def __init__(self, args, logger, num_heads=8, num_trans_layers=6):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        num_heads = args.num_head_encoder
        num_trans_layers = args.num_layer_encoder
        # teacher model
        self.model_teacher = AutoModelForSequenceClassification.from_pretrained(self.config.model_teacher, return_dict=True, output_hidden_states=True)
        # freeze the teacher model
        for name, p in self.model_teacher.named_parameters():
            p.requires_grad = False
        logger.info(f"Loading teacher model:{self.config.model_teacher}")
        # student model
        self.model_student = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        # freeze the encoder
        if args.freeze_teacher==True:
            for name, param in self.model_teacher.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze student model !")
        if args.freeze_student==True:
            for name, param in self.model_student.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze teacher model !")
        self.hidden_dim = self.model_student.config.hidden_size
        # transformer-based attention emb of text and emb of pos
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: StudentAttnTrans")
        self.logger.info(f"hidden_size: {self.hidden_dim}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        self.logger.info(f"TransformerEncoder: {num_heads} num_heads {num_trans_layers} num_trans_layers")
        
    def forward(self, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        # input:
        # batch_user_input_ids and batch_user_attention_mask: {batch, max_length_tweet, max_length_token}
        for idx_user, [user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # emb_tweets_student: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            emb_tweets_student = self.model_student(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            emb_tweets_student = get_pool_emb(emb_tweets_student['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)

            # emb_tweets_teacher: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            with torch.no_grad():
                self.model_teacher.eval()
                emb_tweets_teacher = self.model_teacher(
                    input_ids=user_input_ids, attention_mask=user_attention_mask, 
                )
                emb_tweets_teacher = get_pool_emb(emb_tweets_teacher['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)

            # batch_emb_tweets_student: {batch, max_length_tweet, dim_of_bert}
            batch_emb_tweets_student = emb_tweets_student if idx_user==0 else torch.cat((batch_emb_tweets_student, emb_tweets_student))
            batch_emb_tweets_teacher = emb_tweets_teacher if idx_user==0 else torch.cat((batch_emb_tweets_teacher, emb_tweets_teacher))
            # attention layer
            emb_user_student =  emb_tweets_student
            emb_user_student = self.user_encoder(emb_user_student).squeeze(1)
            scores_attn = torch.softmax(self.attn_ff(emb_user_student).squeeze(-1), -1)
            emb_user_student = scores_attn @ emb_user_student
            batch_emb_user_student = emb_user_student if idx_user==0 else torch.vstack((batch_emb_user_student, emb_user_student))
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout 
        batch_emb_user_student = self.dropout(batch_emb_user_student)
        logits_user = self.classifier(batch_emb_user_student)

        return {
            "logits": logits_user,
            "emb_teacher": batch_emb_tweets_teacher, 
            "emb_student": batch_emb_tweets_student, 
            "scores_attn": batch_scores_attn
        }

class StudentAttnTransPos(nn.Module):
    def __init__(self, args, logger, num_heads=8, num_trans_layers=6):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        num_heads = args.num_head_encoder
        num_trans_layers = args.num_layer_encoder
        # teacher model
        self.model_teacher = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=self.config.model_teacher, return_dict=True, output_hidden_states=True)
        # freeze the teacher model
        for name, p in self.model_teacher.named_parameters():
            p.requires_grad = False
        logger.info(f"Loading teacher model:{self.config.model_teacher}")
        # student model
        self.model_student = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        # freeze the encoder
        if args.freeze_teacher==True:
            for name, param in self.model_teacher.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze student model !")
        if args.freeze_student==True:
            for name, param in self.model_student.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze teacher model !")
        self.hidden_dim = self.model_student.config.hidden_size
        # attn-transformer emb of text and emb of pos
        self.pos_emb = nn.Parameter(torch.Tensor(args.max_length_tweet, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: StudentAttnTransPos")
        self.logger.info(f"hidden_size: {self.hidden_dim}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        self.logger.info(f"TransformerEncoder: {num_heads} num_heads {num_trans_layers} num_trans_layers")
        
    def forward(self, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        # input:
        # batch_user_input_ids and batch_user_attention_mask: {batch, max_length_tweet, max_length_token}
        for idx_user, [user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # emb_tweets_student: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            emb_tweets_student = self.model_student(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            emb_tweets_student = get_pool_emb(emb_tweets_student['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)

            # emb_tweets_teacher: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            with torch.no_grad():
                self.model_teacher.eval()
                emb_tweets_teacher = self.model_teacher(
                    input_ids=user_input_ids, attention_mask=user_attention_mask, 
                )
                emb_tweets_teacher = get_pool_emb(emb_tweets_teacher['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)

            # batch_emb_tweets_student: {batch, max_length_tweet, dim_of_bert}
            batch_emb_tweets_student = emb_tweets_student if idx_user==0 else torch.cat((batch_emb_tweets_student, emb_tweets_student))
            batch_emb_tweets_teacher = emb_tweets_teacher if idx_user==0 else torch.cat((batch_emb_tweets_teacher, emb_tweets_teacher))
            # attention layer
            emb_user_student =  emb_tweets_student + self.pos_emb[-len_real:, ]
            emb_user_student = self.user_encoder(emb_user_student).squeeze(1) # [num_posts, hidden_size]
            scores_attn = torch.softmax(self.attn_ff(emb_user_student).squeeze(-1), -1)
            emb_user_student = scores_attn @ emb_user_student
            batch_emb_user_student = emb_user_student if idx_user==0 else torch.vstack((batch_emb_user_student, emb_user_student))
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout 
        batch_emb_user_student = self.dropout(batch_emb_user_student)
        logits_user = self.classifier(batch_emb_user_student)
        return {
            "logits": logits_user,
            "emb_teacher": batch_emb_tweets_teacher, 
            "emb_student": batch_emb_tweets_student, 
            "scores_attn": batch_scores_attn
        }

class StudentMLP_offline(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        # student model
        self.model_student = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model: {self.config.model_student}")
        # freeze the encoder
        if args.freeze_student==True:
            for name, param in self.model_student.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze teacher model !")
        self.hidden_dim = self.model_student.config.hidden_size
        # clf
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: StudentMLP")
        self.logger.info(f"hidden_size: {self.hidden_dim}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        self.logger.info(f"classifier: mlp")
        
    def forward(self, batch_emb_teacher, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        # input:
        # batch_user_input_ids and batch_user_attention_mask: {batch, max_length_tweet, max_length_token}
        for idx_user, [user_emb_teacher, user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_emb_teacher, batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # emb_tweets_student: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            # batch_emb_tweets_student: {batch, max_length_tweet, dim_of_bert}
            emb_tweets_student = self.model_student(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            emb_tweets_student = get_pool_emb(emb_tweets_student['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)
            batch_emb_tweets_student = emb_tweets_student if idx_user==0 else torch.cat((batch_emb_tweets_student, emb_tweets_student))
            # emb_tweets_teacher: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            emb_tweets_teacher = user_emb_teacher[:len_real,]
            batch_emb_tweets_teacher = emb_tweets_teacher if idx_user==0 else torch.cat((batch_emb_tweets_teacher, emb_tweets_teacher))
            # mean pooling of each tweet: emb_user_student: {dim_of_bert}
            emb_user_student = torch.mean(emb_tweets_student, dim=0)
            # emb_user_student = torch.mean(emb_tweets_student, dim=0)
            batch_emb_user_student = emb_user_student if idx_user==0 else torch.vstack((batch_emb_user_student, emb_user_student))
        # dropout 
        batch_emb_user_student = self.dropout(batch_emb_user_student)
        logits_user = self.classifier(batch_emb_user_student)

        return {
            "logits": logits_user,
            "emb_teacher": batch_emb_tweets_teacher, 
            "emb_student": batch_emb_tweets_student, 
        }


class StudentAttn_offline(nn.Module):
    def __init__(self, args, logger):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        # student model
        self.model_student = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        self.hidden_dim = self.model_student.config.hidden_size
        # freeze the encoder
        if args.freeze_student==True:
            for name, param in self.model_student.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze teacher model !")
        # attn
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: StudentAttn")
        self.logger.info(f"hidden_size: {self.hidden_dim}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        
    def forward(self, batch_emb_teacher, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        # input:
        # batch_user_input_ids and batch_user_attention_mask: {batch, max_length_tweet, max_length_token}
        for idx_user, [user_emb_teacher, user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_emb_teacher, batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # emb_tweets_student: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            # batch_emb_tweets_student: {batch, max_length_tweet, dim_of_bert}
            emb_tweets_student = self.model_student(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            emb_tweets_student = get_pool_emb(emb_tweets_student['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)
            batch_emb_tweets_student = emb_tweets_student if idx_user==0 else torch.cat((batch_emb_tweets_student, emb_tweets_student))
            # emb_tweets_teacher: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            emb_tweets_teacher = user_emb_teacher[:len_real,]
            batch_emb_tweets_teacher = emb_tweets_teacher if idx_user==0 else torch.cat((batch_emb_tweets_teacher, emb_tweets_teacher))
            # attention layer
            scores_attn = torch.softmax(self.attn_ff(emb_tweets_student).squeeze(-1), -1)
            # weighted sum [hidden_size, ]
            emb_user_student = scores_attn @ emb_tweets_student
            batch_emb_user_student = emb_user_student if idx_user==0 else torch.vstack((batch_emb_user_student, emb_user_student))
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout and clf
        batch_emb_user_student = self.dropout(batch_emb_user_student)
        logits_user = self.classifier(batch_emb_user_student)
        return {
            "logits": logits_user,
            "emb_teacher": batch_emb_tweets_teacher, 
            "emb_student": batch_emb_tweets_student, 
            "scores_attn": batch_scores_attn
        }


class StudentAttnTrans_offline(nn.Module):
    def __init__(self, args, logger, num_heads=8, num_trans_layers=6):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        num_heads = args.num_head_encoder
        num_trans_layers = args.num_layer_encoder
        # student model
        self.model_student = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        # freeze the encoder
        if args.freeze_student==True:
            for name, param in self.model_student.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze teacher model !")
        self.hidden_dim = self.model_student.config.hidden_size
        # transformer-based attention emb of text and emb of pos
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: StudentAttnTrans")
        self.logger.info(f"hidden_size: {self.hidden_dim}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        self.logger.info(f"TransformerEncoder: {num_heads} num_heads {num_trans_layers} num_trans_layers")
        
    def forward(self, batch_emb_teacher, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        # input:
        # batch_user_input_ids and batch_user_attention_mask: {batch, max_length_tweet, max_length_token}
        for idx_user, [user_emb_teacher, user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_emb_teacher, batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # emb_tweets_student: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            # batch_emb_tweets_student: {batch, max_length_tweet, dim_of_bert}
            emb_tweets_student = self.model_student(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            emb_tweets_student = get_pool_emb(emb_tweets_student['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)
            batch_emb_tweets_student = emb_tweets_student if idx_user==0 else torch.cat((batch_emb_tweets_student, emb_tweets_student))
            # emb_tweets_teacher: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            emb_tweets_teacher = user_emb_teacher[:len_real,]
            batch_emb_tweets_teacher = emb_tweets_teacher if idx_user==0 else torch.cat((batch_emb_tweets_teacher, emb_tweets_teacher))
            # attention layer
            emb_user_student =  emb_tweets_student
            emb_user_student = self.user_encoder(emb_user_student).squeeze(1)
            scores_attn = torch.softmax(self.attn_ff(emb_user_student).squeeze(-1), -1)
            emb_user_student = scores_attn @ emb_user_student
            batch_emb_user_student = emb_user_student if idx_user==0 else torch.vstack((batch_emb_user_student, emb_user_student))
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout 
        batch_emb_user_student = self.dropout(batch_emb_user_student)
        logits_user = self.classifier(batch_emb_user_student)

        return {
            "logits": logits_user,
            "emb_teacher": batch_emb_tweets_teacher, 
            "emb_student": batch_emb_tweets_student, 
            "scores_attn": batch_scores_attn
        }

class StudentAttnTransPos_offline(nn.Module):
    def __init__(self, args, logger, num_heads=8, num_trans_layers=6):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        num_heads = args.num_head_encoder
        num_trans_layers = args.num_layer_encoder
        # student model
        self.model_student = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        # freeze the encoder
        if args.freeze_student==True:
            for name, param in self.model_student.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze teacher model !")
        self.hidden_dim = self.model_student.config.hidden_size
        # attn-transformer emb of text and emb of pos
        self.pos_emb = nn.Parameter(torch.Tensor(args.max_length_tweet, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: StudentAttnTransPos")
        self.logger.info(f"hidden_size: {self.hidden_dim}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        self.logger.info(f"TransformerEncoder: {num_heads} num_heads {num_trans_layers} num_trans_layers")
        
    def forward(self, batch_emb_teacher, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        # input:
        # batch_user_input_ids and batch_user_attention_mask: {batch, max_length_tweet, max_length_token}
        for idx_user, [user_emb_teacher, user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_emb_teacher, batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            # emb_tweets_student: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            # batch_emb_tweets_student: {batch, max_length_tweet, dim_of_bert}
            emb_tweets_student = self.model_student(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            emb_tweets_student = get_pool_emb(emb_tweets_student['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)
            batch_emb_tweets_student = emb_tweets_student if idx_user==0 else torch.cat((batch_emb_tweets_student, emb_tweets_student))
            # emb_tweets_teacher: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            emb_tweets_teacher = user_emb_teacher[:len_real,]
            batch_emb_tweets_teacher = emb_tweets_teacher if idx_user==0 else torch.cat((batch_emb_tweets_teacher, emb_tweets_teacher))
            # attention layer
            emb_user_student =  emb_tweets_student + self.pos_emb[-len_real:, ]
            emb_user_student = self.user_encoder(emb_user_student).squeeze(1) # [num_posts, hidden_size]
            scores_attn = torch.softmax(self.attn_ff(emb_user_student).squeeze(-1), -1)
            emb_user_student = scores_attn @ emb_user_student
            batch_emb_user_student = emb_user_student if idx_user==0 else torch.vstack((batch_emb_user_student, emb_user_student))
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout 
        batch_emb_user_student = self.dropout(batch_emb_user_student)
        logits_user = self.classifier(batch_emb_user_student)
        return {
            "logits": logits_user,
            "emb_teacher": batch_emb_tweets_teacher, 
            "emb_student": batch_emb_tweets_student, 
            "scores_attn": batch_scores_attn
        }

class MoodAndContent(nn.Module):
    def __init__(self, args, logger, num_heads=8, num_trans_layers=6):
        super().__init__()
        self.config = args
        self.logger = logger
        self.pool_strategy = self.config.pool_strategy
        num_heads = args.num_head_encoder
        num_trans_layers = args.num_layer_encoder
        # student model
        self.model_student = AutoModel.from_pretrained(self.config.model_student, return_dict=True, output_hidden_states=True)
        logger.info(f"Loading student model:{self.config.model_student}")
        # freeze the encoder
        if args.freeze_student==True:
            for name, param in self.model_student.named_parameters():
                param.requires_grad = False
            logger.info(f"Freeze teacher model !")
        self.hidden_dim = self.model_student.config.hidden_size * 2 
        # attn-transformer emb of text and emb of pos
        self.pos_emb = nn.Parameter(torch.Tensor(args.max_length_tweet, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        # mapping of emb of his
        self.linear_his = nn.Linear(in_features=73, out_features=73)
        # classifier
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=2)
        # some config
        self.logger.info(f"core model: StudentAttnTransPos")
        self.logger.info(f"hidden_size: {self.hidden_dim}")
        self.logger.info(f"pool_strategy: {self.pool_strategy}")
        self.logger.info(f"TransformerEncoder: {num_heads} num_heads {num_trans_layers} num_trans_layers")
        
    def forward(self, batch_emb_teacher, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        # input:
        # batch_user_input_ids and batch_user_attention_mask: {batch, max_length_tweet, max_length_token}
        for idx_user, [user_emb_teacher, user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_emb_teacher, batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            emb_tweets_teacher = user_emb_teacher[:len_real,]
            # emb_tweets_student: extract the pooled emb of last layer {max_length_tweet, dim_of_bert}
            # batch_emb_tweets_student: {batch, max_length_tweet, dim_of_bert}
            emb_tweets_student = self.model_student(
                input_ids=user_input_ids, attention_mask=user_attention_mask,
            )
            emb_tweets_student = get_pool_emb(emb_tweets_student['hidden_states'], user_attention_mask, pool_strategy=self.pool_strategy)
            # merge mood and content
            emb_tweets_student = torch.hstack((emb_tweets_student, emb_tweets_teacher))
            # attention layer
            emb_user_student =  emb_tweets_student + self.pos_emb[-len_real:, ]
            emb_user_student = self.user_encoder(emb_user_student).squeeze(1) # [num_posts, hidden_size]
            scores_attn = torch.softmax(self.attn_ff(emb_user_student).squeeze(-1), -1)
            emb_user_student = scores_attn @ emb_user_student
            batch_emb_user_student = emb_user_student if idx_user==0 else torch.vstack((batch_emb_user_student, emb_user_student))
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # dropout 
        batch_emb_user_student = self.dropout(batch_emb_user_student)
        logits_user = self.classifier(batch_emb_user_student)
        return {
            "logits": logits_user,
            "scores_attn": batch_scores_attn
        }

class GRUAttnModel(nn.Module):
    def __init__(self, emb_size, hidden_size, attn_size, gru_layers=1, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.gru_layers = gru_layers
        self.word_rnn = nn.GRU(emb_size, hidden_size, gru_layers, bidirectional=True, batch_first=True)
        self.word_attention = nn.Linear(2*hidden_size, attn_size)
        self.word_context_vector = nn.Linear(attn_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, embedded, attention_mask=None):
        if attention_mask is not None:
            words_per_sentence = attention_mask.sum(1).tolist()
        else:
            words_per_sentence = [embedded.shape[1]] * embedded.shape[0]
        # Re-arrange as words by removing word-pads (SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(embedded,
                                            lengths=words_per_sentence,
                                            batch_first=True,
                                            enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened words (n_words, word_emb)

        # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_words, _ = self.word_rnn(
            packed_words)  # a PackedSequence object, where 'data' is the output of the RNN (n_words, 2 * word_rnn_size)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.word_attention(packed_words.data)  # (n_words, att_size)
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        att_w, _ = pad_packed_sequence(PackedSequence(data=att_w,
                                                      batch_sizes=packed_words.batch_sizes,
                                                      sorted_indices=packed_words.sorted_indices,
                                                      unsorted_indices=packed_words.unsorted_indices),
                                       batch_first=True)  # (n_sentences, max(words_per_sentence))

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(packed_words,
                                           batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)
        # print(f"att_w: {att_w.shape}, word_alphas: {word_alphas.shape}, word_alphas.unsqueeze(2): {word_alphas.unsqueeze(2).shape}, sentences: {sentences.shape}")
        word_alphas = word_alphas.squeeze()
        if word_alphas.dim()==1:
            # attention score of each sentence in passage
            return sentences, word_alphas
        else:
            # attention score of each word in sentence
            return sentences

class GRUHANClassifier(nn.Module):
    def __init__(self, args, logger, vocab_size=30522, emb_size=100, hidden_size=100, attn_size=100,
                 gru_layers=1, dropout=0.1):
        super().__init__()
        self.config = args
        self.logger = logger
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.gru_layers = gru_layers

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        # self.emb = nn.Embedding(vocab_size, emb_size)
        self.post_encoder = GRUAttnModel(emb_size, hidden_size, attn_size, gru_layers, dropout)
        self.user_encoder = GRUAttnModel(2*hidden_size, hidden_size, attn_size, gru_layers, dropout)
        self.clf = nn.Linear(in_features=2*hidden_size, out_features=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_user_input_ids, batch_user_attention_mask, batch_len_real):
        for idx_user, [user_input_ids, user_attention_mask, len_real] in enumerate(zip(batch_user_input_ids, batch_user_attention_mask, batch_len_real)):
            user_input_ids, user_attention_mask = user_input_ids[:len_real, ], user_attention_mask[:len_real, ]
            embedded = self.emb(user_input_ids)
            # [num_posts, seq_len, emb_size] -> [num_posts, 2*hidden_size]:
            x = self.post_encoder(embedded, user_attention_mask).unsqueeze(0)
            post_attention_mask = (user_attention_mask.sum(1) > 2).float().unsqueeze(0)
            feat, scores_attn = self.user_encoder(x, post_attention_mask)
            feat = feat.view(-1) # [2*hidden_size, ]
            batch_feat = feat if idx_user==0 else torch.vstack((batch_feat, feat))
            if len(scores_attn)!=self.config.max_length_tweet:
                score_pad = torch.zeros((self.config.max_length_tweet-len(scores_attn)), dtype=torch.float32).to(scores_attn.device)
                scores_attn = torch.cat((scores_attn, score_pad))
            batch_scores_attn = scores_attn if idx_user==0 else torch.vstack((batch_scores_attn, scores_attn))
        # batch_scores_attn = torch.zeros(self.config.max_length_tweet, dtype=torch.float32).to(batch_feat.device)
        x = self.dropout(batch_feat)
        logits = self.clf(x).squeeze(-1)
        # print(f"logits:{logits.shape}, batch_scores_attn:{batch_scores_attn.shape}")
        return {"logits": logits, "scores_attn": batch_scores_attn}