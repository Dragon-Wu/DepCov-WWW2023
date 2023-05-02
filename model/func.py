import os
from pathlib import Path
import random
import collections
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.metrics import classification_report

from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, ExponentialLR, OneCycleLR, CyclicLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(hidden_state_last, attention_mask):
    # First element of model_output contains all token embeddings
    # hidden_state_last = hidden_state_last[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(hidden_state_last.size()).float()
    )
    sum_embeddings = torch.sum(hidden_state_last * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def emb_mean_pooling_user(emb_all, num_per_record, num_user):
    for idx in range(num_user):
        idx_low, idx_high = idx * num_per_record, (idx + 1) * num_per_record
        emb_user_one = emb_all[
            idx_low:idx_high,
        ]
        emb_user_one = torch.mean(emb_user_one, dim=0)
        emb_user_all = (
            emb_user_one if idx == 0 else torch.vstack((emb_user_all, emb_user_one))
        )
    return emb_user_all


def input_flatten(tweet_input_ids, num_sample=28):
    for idx, tweet_input_ids in enumerate(tweet_input_ids):
        tweet_input_ids = tweet_input_ids[:num_sample]
        user_input_ids_flatten = (
            torch.cat((user_input_ids_flatten, tweet_input_ids))
            if idx != 0
            else tweet_input_ids
        )
    return user_input_ids_flatten


def logits_to_pred_prob_probofpos(logits):
    '''
    binary classification
    input: logits
    output: prob, pred, prob_positive
    '''
    logits = logits.cpu()
    prob = nn.Softmax(dim=1)(logits)
    pred = torch.argmax(prob, axis=1)
    prob_positive = prob[:, 1]
    return prob, pred, prob_positive


def cal_metric(y_true, y_pred, y_prob, name_performance="Training", logger=None):
    """
    performance(y_true, y_pred, y_prob, name_performance='Training')
    """
    acc = accuracy_score(y_true, y_pred)
    f_score = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    if logger:
        logger.info(classification_report(y_true, y_pred))
        logger.info(
            f"{name_performance} f1: {f_score:.4f} acc: {acc:.4f} auroc: {auroc:.4f} ap: {auprc:.4f} "
        )
    else:
        print("\n", classification_report(y_true, y_pred))
        print(
            f"{name_performance} f1: {f_score:.4f} acc: {acc:.4f} auroc: {auroc:.4f} ap: {auprc:.4f} "
        )
    return acc, f_score, auroc, auprc


def cal_loss_clf(criterion_clf, logits, labels):
    loss_clf = criterion_clf(logits, labels)
    return loss_clf


def cal_loss_disitll_clf(
    criterion_distill,
    criterion_clf,
    emb_teacher,
    emb_student,
    logits,
    labels,
    weight_distill,
    weight_clf,
):
    loss_distill = criterion_distill(emb_teacher, emb_student) * weight_distill
    loss_clf = criterion_clf(logits, labels) * weight_clf
    loss = loss_distill + loss_clf
    return loss, loss_distill, loss_clf


def save_checkpoint(output_dir, model, idx_epoch, idx_step, best_f1, best_acc, logger):
    # remove before
    for f in Path(output_dir).glob(f"epoch_*.pt"):
        os.remove(f)
    output_dir = os.path.join(output_dir, f'epoch_{idx_epoch}-step_{idx_step}-best_f1_{best_f1:.4}-best_acc_{best_acc:.4f}.pt')
    torch.save(model.state_dict(), output_dir)
    if logger:
        logger.info(f'Saving models checkpoint to {output_dir}')
    else:
        print(f'Saving models checkpoint to {output_dir}')

# from: https://github.com/Tongjilibo/bert4torch
def get_pool_emb(hidden_state=None, attention_mask=None, pool_strategy='cls', custom_layer=None):
    ''' 获取句向量
    '''
    if pool_strategy == 'cls':
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} strategy request tensor hidden_state'
        return hidden_state[:, 0]
    elif pool_strategy in {'last-avg', 'mean'}:
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} pooling strategy request tensor hidden_state'
        hid = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / attention_mask
    elif pool_strategy in {'last-max', 'max'}:
        if isinstance(hidden_state, (list, tuple)):
            hidden_state = hidden_state[-1]
        assert isinstance(hidden_state, torch.Tensor), f'{pool_strategy} pooling strategy request tensor hidden_state'
        hid = hidden_state * attention_mask[:, :, None]
        return torch.max(hid, dim=1)
    elif pool_strategy == 'first-last-avg':
        assert isinstance(hidden_state, list), f'{pool_strategy} pooling strategy request list hidden_state'
        hid = torch.sum(hidden_state[1] * attention_mask[:, :, None], dim=1) # 这里不取0
        hid += torch.sum(hidden_state[-1] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / (2 * attention_mask)
    elif pool_strategy == 'custom':
        # 取指定层
        assert isinstance(hidden_state, list), f'{pool_strategy} pooling strategy request list hidden_state'
        assert isinstance(custom_layer, (int, list, tuple)), f'{pool_strategy} pooling strategy request int/list/tuple custom_layer'
        custom_layer = [custom_layer] if isinstance(custom_layer, int) else custom_layer
        hid = 0
        for i, layer in enumerate(custom_layer, start=1):
            hid += torch.sum(hidden_state[layer] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / (i * attention_mask)
    else:
        raise ValueError('pool_strategy illegal')

def get_scheduler_batch(optimizer, mode, lr_base, num_epoch, num_batch, **kwargs):
    # all step
    num_training_steps = num_epoch * num_batch
    if mode == "StepLR":
        num_step_decay = num_training_steps*0.2
        scheduler = StepLR(optimizer, step_size=num_step_decay, gamma=0.5)
    elif mode == "MultiStepLR":
        list_milestones = [ num_training_steps*rate for rate in [0.2, 0.5, 0.7, 0.8, 0.9] ]  
        scheduler = MultiStepLR(optimizer, milestones=list_milestones, gamma=0.8)
    elif mode == "LambdaLR":
        scheduler = LambdaLR(
            optimizer, lr_lambda=lambda batch: 1 / (batch + 1)
        )
    elif mode == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=0.998)
    elif mode == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr_base,
            total_steps= num_training_steps,
        )
    elif mode == "CyclicLR":
        scheduler = CyclicLR(
            optimizer,
            base_lr=lr_base / 4.0,
            max_lr=lr_base,
            step_size_up= num_epoch/6*num_batch,
            cycle_momentum=False,
        )
    elif mode == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=num_training_steps//7, eta_min=2e-7
        )
    elif mode == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(num_training_steps*0.1), T_mult=3)
    elif mode == "get_linear_schedule_with_warmup":
        num_warmup_steps = num_training_steps * 0.1
        # kwargs['warmup_proportion']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif mode == "get_cosine_schedule_with_warmup":
        num_warmup_steps = num_training_steps * 0.1
        # kwargs['warmup_proportion']
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
        )
    elif mode == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    return scheduler

def text_segmentate(text, maxlen, seps='\n', strips=None, truncate=True):
    """将文本按照标点符号划分为若干个短句
       truncate: True表示标点符号切分后仍然超长时, 按照maxlen硬截断分成若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips, truncate))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips, truncate))
        return texts
    elif truncate and (not seps) and (len(text) > maxlen):
        # 标点符号用完，仍然超长，且设置了truncate=True
        return [text[i*maxlen:(i+1)*maxlen] for i in range(0, int(np.ceil(len(text)/maxlen)))]
    else:
        return [text]

def merge_segmentate(sequences, maxlen, sep=''):
    '''把m个句子合并成不超过maxlen的n个句子, 主要用途是合并碎句子
    '''
    sequences_new = []
    text = ''
    for t in sequences:
        if text and len(text + sep + t) <= maxlen:
            text = text + sep + t
        elif text:
            sequences_new.append(text)
            text = t
        elif len(t) < maxlen: # text为空
            text = t
        else:
            sequences_new.append(t)
            text = ''
    if text:
        sequences_new.append(text)
    return 

def text_augmentation(texts, noise_dict=None, noise_len=0, noise_p=0.0, skip_words=None, strategy='random', allow_dup=True):
    '''简单的EDA策略, 增删改
    texts: 需要增强的文本/文本list
    noise_dict: 噪音数据, 元素为str的list, tuple, set
    noise_len: 噪音长度, 优先试用
    noise_p: 噪音比例
    skip_words: 跳过的短语, string/list
    strategy: 修改的策略, 包含增insert, 删delete, 改replace, 随机random
    allow_dup: 是否允许同一个位置多次EDA
    '''
    def insert(text, insert_idx, noise_dict):
        text = list(text)
        for i in insert_idx:
            text[i] = text[i] + random.choice(noise_dict)
        return ''.join(text)

    def delete(text, delete_idx):
        text = list(text)
        for i in delete_idx:
            text[i] = ''
        return ''.join(text)

    def replace(text, replace_idx, noise_dict):
        text = list(text)
        for i in replace_idx:
            text[i] = random.choice(noise_dict)
        return ''.join(text)

    def search(pattern, sequence, keep_last=True):
        """从sequence中寻找子串pattern, 返回符合pattern的id集合
        """
        n = len(pattern)
        pattern_idx_set = set()
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                pattern_idx_set = pattern_idx_set.union(set(range(i, i+n))) if keep_last else pattern_idx_set.union(set(range(i, i+n-1)))
        return pattern_idx_set

    if (noise_len==0) and (noise_p==0):
        return texts

    assert strategy in {'insert', 'delete', 'replace', 'random'}, 'EDA strategy only support insert, delete, replace, random'

    if isinstance(texts, str):
        texts = [texts]

    if skip_words is None:
        skip_words = []
    elif isinstance(skip_words, str):
        skip_words = [skip_words]

    for id, text in enumerate(texts):
        sel_len = noise_len if noise_len > 0 else int(len(text)*noise_p) # 噪声长度
        skip_idx = set()  # 不能修改的idx区间
        for item in skip_words:
            # insert时最后一位允许插入
            skip_idx = skip_idx.union(search(item, text, strategy!='insert'))

        sel_idxs = [i for i in range(len(text)) if i not in skip_idx]  # 可供选择的idx区间
        sel_len = sel_len if allow_dup else min(sel_len, len(sel_idxs))  # 无重复抽样需要抽样数小于总样本
        if (sel_len == 0) or (len(sel_idxs) == 0):  # 如果不可采样则跳过
            continue
        sel_idx = np.random.choice(sel_idxs, sel_len, replace=allow_dup)
        if strategy == 'insert':
            texts[id] = insert(text, sel_idx, noise_dict)
        elif strategy == 'delete':
            texts[id] = delete(text, sel_idx)
        elif strategy == 'replace':
            texts[id] = replace(text, sel_idx, noise_dict)
        elif strategy == 'random':
            if random.random() < 0.333:
                skip_idx = set()  # 不能修改的idx区间
                for item in skip_words:
                    # insert时最后一位允许插入
                    skip_idx = skip_idx.union(search(item, text, keep_last=False))
                texts[id] = insert(text, sel_idx, noise_dict)
            elif random.random() < 0.667:
                texts[id] = delete(text, sel_idx)
            else:
                texts[id] = replace(text, sel_idx, noise_dict)
    return texts if len(texts) > 1 else texts[0]

def parallel_apply_generator(func, iterable, workers, max_queue_size, dummy=False, random_seeds=True):
    """多进程或多线程地将func应用到iterable的每个元素中（直接从bert4keras中移植过来）。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是输出可能是func(c), func(a), func(b)。结果将作为一个
    generator返回，其中每个item是输入的序号以及该输入对应的处理结果。
    参数：
        dummy: False是多进程/线性，True则是多线程/线性；
        random_seeds: 每个进程的随机种子。
    """
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import Pool, Queue

    in_queue, out_queue, seed_queue = Queue(max_queue_size), Queue(), Queue()
    if random_seeds is True:
        random_seeds = [None] * workers
    elif random_seeds is None or random_seeds is False:
        random_seeds = []
    for seed in random_seeds:
        seed_queue.put(seed)

    def worker_step(in_queue, out_queue):
        """单步函数包装成循环执行
        """
        if not seed_queue.empty():
            np.random.seed(seed_queue.get())
        while True:
            i, d = in_queue.get()
            r = func(d)
            out_queue.put((i, r))

    # 启动多进程/线程
    pool = Pool(workers, worker_step, (in_queue, out_queue))

    # 存入数据，取出结果
    in_count, out_count = 0, 0
    for i, d in enumerate(iterable):
        in_count += 1
        while True:
            try:
                in_queue.put((i, d), block=False)
                break
            except six.moves.queue.Full:
                while out_queue.qsize() > max_queue_size:
                    yield out_queue.get()
                    out_count += 1
        if out_queue.qsize() > 0:
            yield out_queue.get()
            out_count += 1

    while out_count != in_count:
        yield out_queue.get()
        out_count += 1

    pool.terminate()


def parallel_apply(func, iterable, workers, max_queue_size, callback=None, dummy=False, random_seeds=True, unordered=True):
    """多进程或多线程地将func应用到iterable的每个元素中（直接从bert4keras中移植过来）。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是输出可能是func(c), func(a), func(b)。
    参数：
        callback: 处理单个输出的回调函数；
        dummy: False是多进程/线性，True则是多线程/线性；windows需设置dummy=True
        random_seeds: 每个进程的随机种子；
        unordered: 若为False，则按照输入顺序返回，仅当callback为None时生效。
    """
    generator = parallel_apply_generator(func, iterable, workers, max_queue_size, dummy, random_seeds)

    if callback is None:
        if unordered:
            return [d for i, d in generator]
        else:
            results = sorted(generator, key=lambda d: d[0])
            return [d for i, d in results]
    else:
        for i, d in generator:
            callback(d)


class ContrastiveLoss(nn.Module):
    """对比损失：减小正例之间的距离，增大正例和反例之间的距离
    公式：labels * distance_matrix.pow(2) + (1-labels)*F.relu(margin-distance_matrix).pow(2)
    https://www.sbert.net/docs/package_reference/losses.html
    """
    def __init__(self, margin=0.5, size_average=True, online=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.online = online

    def forward(self, distances, labels, pos_id=1, neg_id=0):
        if not self.online:
            losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
            return losses.mean() if self.size_average else losses.sum()
        else:
            negs = distances[labels == neg_id]
            poss = distances[labels == pos_id]

            # select hard positive and hard negative pairs
            negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]
            
            positive_loss = positive_pairs.pow(2).sum()
            negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
            return positive_loss + negative_loss

class FGM():
    '''对抗训练
    '''
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings', **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0 and not torch.isnan(norm):  # nan是为了apex混合精度时:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb', **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    '''对抗训练
    '''
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False, **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):  # nan是为了apex混合精度时
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb', **kwargs):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
        
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
        
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            # 修复如pooling层参与foward，但是不参与backward过程时grad为空的问题
            if param.requires_grad and (param.grad is not None):
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (param.grad is not None):
                param.grad = self.grad_backup[name]


class VAT():
    '''虚拟对抗训练 https://github.com/namisan/mt-dnn/blob/v0.2/alum/adv_masked_lm.py
    '''
    def __init__(self, model, emb_name='word_embeddings', noise_var=1e-5, noise_gamma=1e-6, adv_step_size=1e-3, 
                 adv_alpha=1, norm_type='l2', **kwargs):
        self.model = model
        self.noise_var = noise_var  # 噪声的方差
        self.noise_gamma = noise_gamma # eps
        self.adv_step_size = adv_step_size  # 学习率
        self.adv_alpha = adv_alpha  # 对抗loss的权重
        self.norm_type = norm_type  # 归一化方式
        self.embed = None
        for (name, module) in self.model.named_modules():
            if emb_name in name:
                module.register_forward_hook(hook=self.hook)

    def hook(self, module, fea_in, fea_out):
        self.embed = fea_out
        return None
    
    def forward_(self, train_X, new_embed):
        # 把原来的train_X中的token_ids换成embedding形式
        if isinstance(train_X, (tuple, list)):
            new_train_X = [new_embed] + train_X[1:]
            adv_output = self.model.forward(*new_train_X) if self.model.forward.__code__.co_argcount >= 3 else self.model.forward(new_train_X)
        elif isinstance(train_X, torch.Tensor):
            adv_output = self.model.forward(new_embed)
        return adv_output

    def virtual_adversarial_training(self, train_X, logits):
        # 初始扰动 r
        noise = self.embed.data.new(self.embed.size()).normal_(0, 1) * self.noise_var
        noise.requires_grad_()
        # x + r
        new_embed = self.embed.data.detach() + noise
        adv_output = self.forward_(train_X, new_embed)  # forward第一次
        adv_logits = adv_output[0] if isinstance(adv_output, (list, tuple)) else adv_output
        adv_loss = self.kl(adv_logits, logits.detach(), reduction="batchmean")
        delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
        norm = delta_grad.norm()
        # 梯度消失，退出
        if torch.isnan(norm) or torch.isinf(norm):
            return None
        # inner sum
        noise = noise + delta_grad * self.adv_step_size
        # projection
        noise = self.adv_project(noise, norm_type=self.norm_type, eps=self.noise_gamma)
        new_embed = self.embed.data.detach() + noise
        new_embed = new_embed.detach()
        # 在进行一次训练
        adv_output = self.forward_(train_X, new_embed)  # forward第二次
        adv_logits = adv_output[0] if isinstance(adv_output, (list, tuple)) else adv_output
        adv_loss_f = self.kl(adv_logits, logits.detach())
        adv_loss_b = self.kl(logits, adv_logits.detach())
        # 在预训练时设置为10，下游任务设置为1
        adv_loss = (adv_loss_f + adv_loss_b) * self.adv_alpha
        return adv_loss
    
    @staticmethod
    def kl(inputs, targets, reduction="sum"):
        """
        计算kl散度
        inputs：tensor，logits
        targets：tensor，logits
        """
        loss = F.kl_div(F.log_softmax(inputs, dim=-1), F.softmax(targets, dim=-1), reduction=reduction)
        return loss

    @staticmethod
    def adv_project(grad, norm_type='inf', eps=1e-6):
        """
        L0,L1,L2正则，对于扰动计算
        """
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction

def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.
    Args:
      inputs: An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    #first = True
    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state

        model_params_keys = list(model_params.keys())
        if params_keys is None: #and first:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                #if not first: continue
                params_dict[k] = p.clone()
                #params_dict[k] = p.clone().zero_()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
                #params_dict[k] = torch.max(torch.cat([p.unsqueeze(dim=0), params_dict[k].unsqueeze(dim=0)], dim=0), dim=0)[0]
        #first = False

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        # averaged_params[k].div_(num_models)
        averaged_params[k] = averaged_params[k]/num_models
    new_state = averaged_params
    return new_state