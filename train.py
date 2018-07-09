import os
import sys
import time
import shutil
import pickle
import argparse
import traceback

import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from generic_utils import (remove_experiment_folder,
                           create_experiment_folder, save_checkpoint,
                           save_best_model, load_config, lr_decay,
                           count_parameters, check_update, get_commit_hash)
from model import FFTNetModel
from model import MaskedCrossEntropyLoss
from model import EMA
from dataset import LJSpeechDataset


def train(epoch):
    avg_loss = 0.0
    epoch_time = 0
    # progbar = Progbar(len(train_loader.dataset) // c.batch_size)
    num_iter_epoch = len(train_loader.dataset) // c.batch_size
    if c.ema_decay > 0:
        ema = EMA(c.ema_decay)
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param)
    else:
        ema = None
    model.train()
    for num_iter, batch in enumerate(train_loader):
        start_time = time.time()
        wav = batch[0].unsqueeze(1)
        mel = batch[1].transpose(1, 2)
        lens = batch[2]
        target = batch[3]
        if use_cuda:
            wav = wav.cuda()
            mel = mel.cuda()
            target = target.cuda()
        current_step = num_iter + epoch * len(train_loader) + 1
        lr = lr_decay(c.lr, current_step, c.warmup_steps)
        for params_group in optimizer.param_groups:
            params_group['lr'] = lr
        optimizer.zero_grad()
        out = torch.nn.parallel.data_parallel(model, (wav, mel))
        # out = model(wav, mel)
        loss, fp, tp = criterion(out, target, lens)
        loss.backward()
        grad_norm, skip_flag = check_update(model, 5, 100)
        if skip_flag:
            optimizer.zero_grad()
            print(" | > Iteration skipped!!")
            continue
        optimizer.step()
        # model ema
        if ema is not None:
            for name, param in model.named_parameters():
                if name in ema.shadow:
                    ema.update(name, param.data)
        step_time = time.time() - start_time
        epoch_time += step_time
        # update
        # progbar.update(num_iter+1, values=[('total_loss', loss.item()),
        #                                    ('grad_norm', grad_norm.item()),
        #                                    ('fp', fp),
        #                                    ('tp', tp)
        #                                   ])
        if current_step % c.print_iter == 0:
            print(" | > loss:{:.4f}\tgrad_norm:{:.4f}\tfp:{}\ttp:{}\tlr:{:.5f}\t".format(loss.item(), grad_norm, fp, tp, params_group['lr']))
        avg_loss += loss.item()
    avg_loss /= num_iter
    return ema, avg_loss


def evaluate(epoch, ema):
    avg_loss = 0.0
    epoch_time = 0
    progbar = Progbar(len(val_loader.dataset) // c.eval_batch_size)
    ema_model = FFTNetModel(hid_channels=256, out_channels=256, n_layers=c.num_quant,
                            cond_channels=80)
    ema_model = ema.assign_ema_model(model, ema_model, use_cuda)
    ema_model.eval()
    with torch.no_grad():
        for num_iter, batch in enumerate(train_loader):
            start_time = time.time()
            wav = batch[0].unsqueeze(1)
            mel = batch[1].transpose(1, 2)
            lens = batch[2]
            target = batch[3]
            if use_cuda:
                wav = wav.cuda()
                mel = mel.cuda()
                target = target.cuda()
            current_step = num_iter + epoch * len(train_loader) + 1
            out = ema_model(wav, mel)
            loss, fp, tp = criterion(out, target, lens)
            step_time = time.time() - start_time
            epoch_time += step_time
            # update
            # progbar.update(num_iter+1, values=[('total_loss', loss.item()),
            #                                    ('fp', fp),
            #                                    ('tp', tp)
            #                                   ])
            avg_loss += loss.item()
    avg_loss /= num_iter
    return avg_loss


def main(args):
    for epoch in range(c.epochs):
        print(" > Epoch: ", epoch)
        ema, avg_loss = train(epoch)
        avg_val_loss = evaluate(epoch, ema)
        print(" -- loss:{:.5f}\tval_loss:{:.5f}".format(avg_loss, avg_val_loss))

if __name__ == "__main__":

    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        help='path to config file for training',)
    parser.add_argument('--debug', type=bool, default=False,
                        help='Stop asking for git hash before the run.')
    parser.add_argument('--finetune_path', type=str)
    args = parser.parse_args()
    c = load_config(args.config_path)

    # setup output paths and read configs
    _ = os.path.dirname(os.path.realpath(__file__))
    OUT_PATH = os.path.join(_, c.output_path)
    OUT_PATH = create_experiment_folder(OUT_PATH, c.model_name, True)
    CHECKPOINT_PATH = os.path.join(OUT_PATH, 'checkpoints')
    shutil.copyfile(args.config_path, os.path.join(OUT_PATH, 'config.json'))

    # setup tensorboard
    tb = SummaryWriter(OUT_PATH)

    model = FFTNetModel(hid_channels=256, out_channels=256, n_layers=c.num_quant,
                        cond_channels=80)
    criterion = MaskedCrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)
    num_params = count_parameters(model)
    print(" > Models has {} parameters".format(num_params))

    if use_cuda:
        model.cuda()
        criterion.cuda()

    train_dataset = LJSpeechDataset(os.path.join(c.data_path, "mels",
                                                 "meta_fftnet_train.csv"),
                              os.path.join(c.data_path, "mels"),
                              c.sample_rate,
                              c.num_mels, c.num_freq,
                              c.min_level_db, c.frame_shift_ms,
                              c.frame_length_ms, c.preemphasis, c.ref_level_db,
                              c.num_quant, c.min_wav_len, c.max_wav_len, False)

    val_dataset =  LJSpeechDataset(os.path.join(c.data_path, "mels",
                                               "meta_fftnet_val.csv"),
                                 os.path.join(c.data_path, "mels"),
                                 c.sample_rate,
                                 c.num_mels, c.num_freq,
                                 c.min_level_db, c.frame_shift_ms,
                                 c.frame_length_ms, c.preemphasis,
                                 c.ref_level_db, c.num_quant, c.min_wav_len,
                                 c.max_wav_len, False)


    train_loader = DataLoader(train_dataset, batch_size=c.batch_size,
                            shuffle=False, collate_fn=train_dataset.collate_fn,
                            drop_last=True, num_workers=c.num_loader_workers)

    val_loader = DataLoader(val_dataset, batch_size=c.eval_batch_size,
                            shuffle=False, collate_fn=train_dataset.collate_fn,
                            drop_last=True, num_workers=4)
    try:
        main(args)
        remove_experiment_folder(OUT_PATH)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception:
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
