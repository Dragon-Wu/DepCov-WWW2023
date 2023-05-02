from model.trainer_pl import trainer_DCT, trainer_DCT_self, trainer_DCT_emb_tweet, trainer_DCT_emb_merge
from data.dataset_pl import Dataloader_DCT, Dataloader_DCT_emb_tweet
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    DeviceStatsMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from model.model import *
from utils.tool_simple import seed_everything, init_logger
import os
import sys
import shutil
import argparse
import datetime
# self-defined method
sys.path.append("..")
# pl

# env
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dataset and model for different model
CLASS_MODEL_DATASET_TRAINER = {
    # Only content or mood:
    'SelfMLP': (SelfMLP, Dataloader_DCT, trainer_DCT_self),
    'SelfAttn': (SelfAttn, Dataloader_DCT, trainer_DCT_self),
    'SelfTrans': (SelfTrans, Dataloader_DCT, trainer_DCT_self),
    'SelfTransPos': (SelfTransPos, Dataloader_DCT, trainer_DCT_self),
    # GRU
    'GRUHANClassifier': (GRUHANClassifier, Dataloader_DCT, trainer_DCT_self),
    # Mood And Content
    'MoodAndContent': (MoodAndContent, Dataloader_DCT_emb_tweet, trainer_DCT_emb_merge),
    # Mood2Content
    'StudentMLP': (StudentMLP, Dataloader_DCT, trainer_DCT),
    'StudentAttn': (StudentAttn, Dataloader_DCT, trainer_DCT),
    'StudentAttnTrans': (StudentAttnTrans, Dataloader_DCT, trainer_DCT),
    'StudentAttnTransPos': (StudentAttnTransPos, Dataloader_DCT, trainer_DCT),
    # Mood2Content and teacher emb is offline (previously obtained)
    'StudentMLP_offline': (StudentMLP_offline, Dataloader_DCT_emb_tweet, trainer_DCT_emb_tweet),
    'StudentAttn_offline': (StudentAttn_offline, Dataloader_DCT_emb_tweet, trainer_DCT_emb_tweet),
    'StudentAttnTrans_offline': (StudentAttnTrans_offline, Dataloader_DCT_emb_tweet, trainer_DCT_emb_tweet),
    'StudentAttnTransPos_offline': (StudentAttnTransPos_offline, Dataloader_DCT_emb_tweet, trainer_DCT_emb_tweet),

}


def main_pl():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument("--mode", default=None, type=str, required=True,
                        help="The mode of model")
    parser.add_argument("--description", default=None, type=str, required=True,
                        help="The detailed description of model")
    # dataset
    parser.add_argument("--path_dir_data", default=None, type=str, required=True,
                        help="The path of data directory")
    parser.add_argument("--num_case", default='max', type=str, required=True,
                        help="The max size of positive case")
    parser.add_argument("--tweet_mode", default='date', type=str, required=True,
                        help="get the tweet merged by date or itself")
    parser.add_argument("--max_length_tweet", default=21, type=int, required=True,
                        help="The max tweet of a user")
    parser.add_argument("--max_length_token", default=256, type=int, required=True,
                        help="The max token of a tweet")
    # loss
    parser.add_argument("--weight_distill", default=1, type=float, required=True,
                        help="weight_distill")
    parser.add_argument("--weight_clf", default=1, type=float, required=True,
                        help="weight_clf")
    parser.add_argument("--ratio_weight_dynamic", default=1, type=float, required=True,
                        help="ratio : weight decay of weight of distill")
    parser.add_argument("--weight_distill_min", default=0, type=float, required=True,
                        help="the min weight of distill")
    # models
    parser.add_argument("--model_teacher", default=None, type=str, required=True,
                        help="model_teacher")
    parser.add_argument("--model_student", default=None, type=str, required=True,
                        help="model_student")
    parser.add_argument("--pool_strategy", default='mean', type=str, required=True,
                        help="pool_strategy")
    parser.add_argument("--freeze_teacher", default=True,
                        help="freeze encoder of teacher")
    parser.add_argument("--freeze_student", default=False,
                        help="freeze encoder of student")

    # encoder setting for emb model
    parser.add_argument("--model_encoder", default='SenRoberta', type=str,
                        help="name of encoder PLMsr")
    parser.add_argument("--dim_emb", default=768, type=int,
                        help="dim of emb")
    parser.add_argument("--num_head_encoder", default=12, type=int,
                        help="num_head_encoder")
    parser.add_argument("--num_layer_encoder", default=12, type=int,
                        help="num_layer_encoder")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="the prob of dropout")

    # learning
    parser.add_argument("--scheduler", default='get_cosine_schedule_with_warmup', type=str,
                        help="scheduler")
    parser.add_argument("--auto_lr_find", default=False,
                        help="auto_lr_find")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="base learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight_decay for regularization")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="adam_epsilon for AdamW")
    parser.add_argument("--use_swa", default=False, help="use_swa")

    # epoch and batch
    parser.add_argument("--epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batchsize_train", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--batchsize_valid", default=64, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--step_accumulate", default=8, type=int,
                        help="Step for accumulating.")
    parser.add_argument("--step_logging", default=5, type=int,
                        help="Log every X updates steps.")
    parser.add_argument("--step_eval", default=0.1, type=float,
                        help="Evaluation steps: fraction of a epoch")
    parser.add_argument("--earlystop_patience", default=10, type=int,
                        help="earlystop_patience")

    # metric
    parser.add_argument('--metric', type=str,
                        default='val_auroc', help="metric for auroc")
    # seed
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # device param
    parser.add_argument('--device', type=str, default='0',
                        help="the cuda device")

    args = parser.parse_args()

    # set seed
    seed_everything(args.seed)
    pl.seed_everything(args.seed)

    # some complement
    args.device = [int(num) for num in str(args.device)]

    # get the record of save
    path_dir_record = "/home/wjg/status_COVID19/DepCovTwi/record"
    args.path_dir_save = os.path.join(path_dir_record, args.mode)
    if not os.path.exists(args.path_dir_save):
        os.makedirs(args.path_dir_save)
    time_now = str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    time_now = args.model_encoder + '-' + str(args.seed) + '-' + time_now
    args.path_dir_save = os.path.join(args.path_dir_save, time_now)
    if not os.path.exists(args.path_dir_save):
        os.makedirs(args.path_dir_save)
    args.output_dir = args.path_dir_save

    logger = init_logger(os.path.join(args.path_dir_save, f'{args.mode}.log'))

    shutil.copy("run.sh", os.path.join(args.path_dir_save, f'run_backup.sh'))
    logger.info(
        f"run.sh has been backup in {os.path.join(args.path_dir_save, f'run_backup.sh')}")

    for key, value in args.__dict__.items():
        logger.info(f"{key}: {value}")

    # data setting
    data_dct = CLASS_MODEL_DATASET_TRAINER[args.mode][1](
        args=args, logger=logger, num_workers=4)
    data_dct.setup()
    # step
    args.num_batch_train = len(
        data_dct.train_dataloader()) / len(args.device) / args.step_accumulate
    args.num_batch_train = int(args.num_batch_train) + 1
    print(
        f"Training: num_epoch:{args.epochs}, num_batch:{args.num_batch_train}, all:{args.epochs*args.num_batch_train}")

    # callbacks
    checkpoint = ModelCheckpoint(
        filename="{epoch:02d}-{val_auroc:.4f}-{val_f1:.4f}",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        monitor=args.metric,
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor=args.metric, patience=args.earlystop_patience, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # device_stats = DeviceStatsMonitor()
    callbacks = [checkpoint, early_stopping, lr_monitor]
    if args.use_swa == True:
        callbacks.append(StochasticWeightAveraging())

    os.makedirs(os.path.join(args.output_dir, args.model_encoder))
    logger_tb = TensorBoardLogger(
        save_dir=args.output_dir, name=args.model_encoder)

    model_core = CLASS_MODEL_DATASET_TRAINER[args.mode][0](
        args=args, logger=logger)

    # print("hparams.auto_lr_find=", args.auto_lr_find)
    # if args.auto_lr_find:
    #     trainer = pl.Trainer(
    #         accelerator="gpu",
    #         devices=[0],
    #         # callbacks=callbacks,
    #         max_epochs=args.epochs,
    #         min_epochs=1,
    #         accumulate_grad_batches=args.step_accumulate,
    #         val_check_interval=args.step_eval,
    #         gradient_clip_val=1.0,
    #         deterministic=True,
    #         logger=logger_tb,
    #         log_every_n_steps=1,
    #         # profiler="simple",
    #     )
    #     # scale of learning rate
    #     lr_finder = trainer.tuner.lr_find(
    #         model,
    #         datamodule=data_dct,
    #     )
    #     fig = lr_finder.plot(suggest=True)
    #     plt.savefig("lr_find.svg", dpi=300)
    #     lr = lr_finder.suggestion()
    #     print("suggest lr=", lr)
    #     model.hparams.learning_rate = lr
    #     args.learning_rate = lr
    #     del trainer
    #     del model
    #     model = model(args, learning_rate=args.learning_rate)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.device,
        strategy="ddp",
        # "ddp_find_unused_parameters_false"
        callbacks=callbacks,
        max_epochs=args.epochs,
        min_epochs=2,
        accumulate_grad_batches=args.step_accumulate,
        val_check_interval=args.step_eval,
        gradient_clip_val=1.0,
        deterministic=True,
        logger=logger_tb,
        log_every_n_steps=1,
        # profiler="simple",
    )

    model = CLASS_MODEL_DATASET_TRAINER[args.mode][2](
        args=args, model=model_core, data=data_dct)

    # model training
    trainer.fit(model, data_dct)

    # save the best
    logger.info(
        f"best_model_path: {trainer.checkpoint_callback.best_model_path}")
    logger.info(
        f"best_model_score: {trainer.checkpoint_callback.best_model_score}")

    # testing
    test_result = trainer.test(
        model, data_dct.test_dataloader(), ckpt_path='best', verbose=True)
    logger.info(test_result)

    path_file_out = f"out/{args.mode}_{args.model_encoder}_{args.seed}.out"
    if os.path.exists(path_file_out):
        shutil.copy(f"{path_file_out}", os.path.join(
            args.path_dir_save, path_file_out[4:]))
        logger.info(
            f"out has been backup in {os.path.join(args.path_dir_save, path_file_out[4:])}")

    logger.info('All done')


if __name__ == '__main__':
    # main()
    main_pl()
