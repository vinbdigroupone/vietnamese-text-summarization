
import os
import pickle

import torch
from torch import optim
import torch.nn as nn
from torch.cuda.amp import GradScaler

from args import parse_args
from config import get_cfg_defaults
from seed import setup_determinism
from model import SeqToSeq
from utils import load_pickle
from dataset import make_dataloader
from train import train_loop
scaler = GradScaler()

def main(args, cfg):
    device = cfg.DEVICE
    hidden_size = cfg.MODEL.HIDDEN_SIZE

    # Load dataset metadata
    path_train = '../Abstractive_method/train_meta'
    path_eval = '../Abstractive_method/eval_meta'
    mode = 'eval'
    if mode == 'train':
        path = path_train
    elif mode == 'eval':
        path = path_eval
    input_lang, output_lang, pairs, word2index, word2count, index2word, n_words = load_pickle(path)

    # Define model
    model = SeqToSeq(cfg, n_words, hidden_size)

    # # Define optimzer
    # optimizer = optim.SGD(encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE)

    # Define loss
    train_criterion = nn.NLLLoss()
    valid_criterion = nn.NLLLoss()

    model = model.to(device)
    model = nn.DataParallel(model)

    # Load checkpoint
    # model, start_epoch, best_metric = load_checkpoint(args, model)
    # if not cfg.NAME in args.load:
    #     best_metric = 0

    # Create Dataloader
    dataloader = make_dataloader(cfg, pairs, input_lang, output_lang, word2index)

    # Run script
    start_epoch = 0
    if args.mode == 'train':
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHES):

            # Start training loop
            train_loader = dataloader
            train_loss = train_loop(cfg, model, train_loader, train_criterion,
                                    epoch, scaler)
            print(best_metric)
            val_loss, best_metric = valid_model(cfg, model, valid_loader, valid_criterion)

    elif args.model == 'valid':
        valid_model(cfg, model, valid_loader, valid_criterion)

if __name__ == '__main__':
    args = parse_args()
    cfg = get_cfg_defaults()

    # make dirs
    for _dir in ['WEIGHTS', 'OUTPUTS', 'LOGS']:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])

    # seed, run
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)