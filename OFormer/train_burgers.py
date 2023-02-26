import os
import sys
import time
import logging
import shutil
import argparse
from typing import Union

import torch
import numpy as np
from tqdm import tqdm

from functools import partial
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from tensorboardX import SummaryWriter

from einops import rearrange
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset

from loss_fn import rel_loss
from nn_module.encoder_module import Encoder1D
from nn_module.decoder_module import PointWiseDecoder1D, BCDecoder1D
from utils import load_checkpoint, save_checkpoint, ensure_dir, get_arguments

# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)


def build_model(res) -> (Encoder1D, PointWiseDecoder1D):
    # currently they are hard coded
    encoder = Encoder1D(
        2,   # u + x coordinates
        96,
        96,
        4,
        res=res
    )

    decoder = PointWiseDecoder1D(
        96,
        1,
        3,
        scale=2,
        res=res
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder


def make_image_grid(init: torch.Tensor, sequence: torch.Tensor, gt: torch.Tensor, out_path, nrow=8):
    b, n, c = sequence.shape   # c = 1

    init = init.detach().cpu().squeeze(-1).numpy()
    sequence = sequence.detach().cpu().squeeze(-1).numpy()
    gt = gt.detach().cpu().squeeze(-1).numpy()

    fig = plt.figure(figsize=(16., 16.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(b//nrow, nrow),  # creates 8x8 grid of axes
                     )
    x = np.linspace(0, 1, n)

    for ax, im_no in zip(grid, np.arange(b)):
        # Iterating over the grid returns the Axes.
        # ax.plot(x, init[im_no], c='b', alpha=0.2)
        ax.plot(x, sequence[im_no], c='r')
        ax.plot(x, gt[im_no], '--', c='g', alpha=0.8)
        ax.axis('equal')
        ax.axis('off')

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


# copied from Galerkin Transformer
def central_diff(x: torch.Tensor, h):
    # assuming PBC
    # x: (batch, seq_len, feats), h is the step size

    pad_0, pad_1 = x[:, -2:-1], x[:, 1:2]
    x = torch.cat([pad_0, x, pad_1], dim=1)
    x_diff = (x[:, 2:] - x[:, :-2])/2  # f(x+h) - f(x-h) / 2h
    # pad = np.zeros(x_diff.shape[0])

    # return np.c_[pad, x_diff/h, pad]
    return x_diff/h


def pad_pbc(x: torch.Tensor, pos: torch.Tensor, h, pad_ratio=1/128):
    # x: (batch, seq_len, feats), h is the step size
    # assuming x in the order of x-axis [0, 1]
    n = x.shape[1]
    pad_0, pad_1 = x[:, -int(pad_ratio*n)-1:-1], x[:, 1:int(pad_ratio*n)+1]
    offset = np.arange(1, int(pad_ratio*n)+1, dtype=np.float32)*h
    pos_pad_0 = 0 - offset[::-1]
    pos_pad_1 = 1 + offset
    pos_pad_0 = rearrange(torch.as_tensor(pos_pad_0).to(pos.device), 'n -> 1 n 1').repeat([x.shape[0], 1, 1])
    pos_pad_1 = rearrange(torch.as_tensor(pos_pad_1).to(pos.device), 'n -> 1 n 1').repeat([x.shape[0], 1, 1])
    return torch.cat([pad_0, x, pad_1], dim=1), torch.cat([pos_pad_0, pos, pos_pad_1], dim=1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train a PDE Transformer (Operator Transformer)")
    parser = get_arguments(parser)
    opt = parser.parse_args()
    print('Using following options')
    print(opt)

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()

    # add code for datasets

    print('Preparing the data')
    data_path = opt.dataset_path
    ntrain = opt.train_seq_num
    ntest = opt.test_seq_num
    res = opt.resolution

    sub = 2 ** 13 // res  # total grid size divided by the target resolution
    dx = 1./res
    # below code is copied from:
    # https://github.com/zongyi-li/fourier_neural_operator/blob/c13b475dcc9bcd855d959851104b770bbcdd7c79/utilities3.py#L19
    # Data is of the shape (number of samples, grid size)
    data = loadmat(data_path)
    x_data = data['a'][:, ::sub]   # input: u(x, 0)
    y_data = data['u'][:, ::sub]   # solution: u(x, 1)

    print(f'x data shape: {x_data.shape}')
    print(f'y data shape: {y_data.shape}')

    print(f'Data resolution: {x_data.shape[-1]}')
    x_train = x_data[:ntrain, :]   # (num_samples, nx)
    y_train = y_data[:ntrain, :]   # (num_samples, nx)
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    x_train = torch.as_tensor(x_train.reshape(ntrain, res, 1), dtype=torch.float32)
    x_test = torch.as_tensor(x_test.reshape(ntest, res, 1), dtype=torch.float32)
    y_train = torch.as_tensor(y_train.reshape(ntrain, res, 1), dtype=torch.float32)
    y_test = torch.as_tensor(y_test.reshape(ntest, res, 1), dtype=torch.float32)

    gridx = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    gridx = gridx.reshape(1, res, 1)
    if use_cuda:
        gridx = gridx.cuda()

    print(f'x train shape: {x_train.shape}')
    print(f'y train shape: {y_train.shape}')

    print(f'x test shape: {x_test.shape}')
    print(f'y test shape: {y_test.shape}')

    print(f'grid x shape: {gridx.shape}')
    # sys.exit()

    train_dataloader = DataLoader(TensorDataset(x_train, y_train),
                                   batch_size=opt.batch_size,
                                   shuffle=True)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test),
                                  batch_size=opt.batch_size,
                                  shuffle=False)

    # instantiate network
    print('Building network')
    encoder, decoder = build_model(res)
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter()
    checkpoint_dir = os.path.join(opt.log_dir, 'model_ckpt')
    ensure_dir(checkpoint_dir)

    sample_dir = os.path.join(opt.log_dir, 'samples')
    ensure_dir(sample_dir)

    # save option information to the disk
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (opt.log_dir, 'logging_info'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('=======Option used=======')
    for arg in vars(opt):
        logger.info(f'{arg}: {getattr(opt, arg)}')

    # save the py script of models
    script_dir = os.path.join(opt.log_dir, 'script_cache')
    ensure_dir(script_dir)
    shutil.copy('nn_module/__init__.py', script_dir)
    shutil.copy('nn_module/attention_module.py', script_dir)
    shutil.copy('nn_module/cnn_module.py', script_dir)
    shutil.copy('nn_module/encoder_module.py', script_dir)
    shutil.copy('nn_module/decoder_module.py', script_dir)
    shutil.copy('nn_module/fourier_neural_operator.py', script_dir)
    # shutil.copy('nn_module/gnn_module.py', script_dir)
    # shutil.copy('train_burgers.py', opt.log_dir)

    # create optimizers
    enc_optim = torch.optim.Adam(list(encoder.parameters()), lr=opt.lr, weight_decay=1e-4)
    dec_optim = torch.optim.Adam(list(decoder.parameters()), lr=opt.lr, weight_decay=1e-4)

    # enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, opt.iters//10, gamma=0.75, last_epoch=-1)
    # dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, opt.iters//10, gamma=0.75, last_epoch=-1)
    enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=opt.iters,
                              div_factor=1e4,
                              pct_start=0.2,
                              final_div_factor=1e4,
                               )
    dec_scheduler = OneCycleLR(dec_optim, max_lr=opt.lr, total_steps=opt.iters,
                               div_factor=1e4,
                               pct_start=0.2,
                               final_div_factor=1e4,
                               )

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    if opt.resume:
        print(f'Resuming checkpoint from: {opt.path_to_resume}')
        ckpt = load_checkpoint(opt.path_to_resume)  # custom method for loading the last checkpoint
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])

        start_n_iter = ckpt['n_iter']

        enc_optim.load_state_dict(ckpt['enc_optim'])
        dec_optim.load_state_dict(ckpt['dec_optim'])

        enc_scheduler.load_state_dict(ckpt['enc_sched'])
        dec_scheduler.load_state_dict(ckpt['dec_sched'])
        print("last checkpoint restored")

    # now we start the main loop
    n_iter = start_n_iter

    # mixed-precision
    # [encoder, decoder], [enc_optim, dec_optim] = amp.initialize(
    #     [encoder, decoder], [enc_optim, dec_optim], opt_level='O0')


    # for loop going through the dataset
    with tqdm(total=opt.iters) as pbar:
        pbar.update(n_iter)
        train_data_iter = iter(train_dataloader)

        while True:

            encoder.train()
            decoder.train()
            start_time = time.time()

            try:
                data = next(train_data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                del train_data_iter
                train_data_iter = iter(train_dataloader)
                data = next(train_data_iter)

            # data preparation
            x, y = data

            if use_cuda:
                x, y = x.cuda(), y.cuda()

            # standardize
            # data_mean = torch.mean(x, dim=1, keepdim=True)
            # data_std = torch.std(x, dim=1, keepdim=True)
            # x = (x - data_mean) / data_std
            # y = (y - data_mean) / data_std

            input_pos = prop_pos = gridx.repeat([x.shape[0], 1, 1])
           # x, input_pos = pad_pbc(x, input_pos, dx)
            x = torch.cat((x, input_pos), dim=-1)   # concat coordinates as an additional feature

            # randomly create some idx
            #input_idx = torch.as_tensor(np.random.choice(input_pos.shape[1], int(0.95*input_pos.shape[1]), replace=False)).view(1, -1).cuda()
            # prop_idx = torch.as_tensor(np.random.choice(prop_pos.shape[1], int(0.75*prop_pos.shape[1]), replace=False)).view(1, -1).cuda()

            # x = index_points(x, input_idx.repeat([x.shape[0], 1]))
            # input_pos = index_points(input_pos, input_idx.repeat([x.shape[0], 1]))

            # y = index_points(y, prop_idx.repeat([x.shape[0], 1]))
            # prop_pos = index_points(prop_pos, prop_idx.repeat([x.shape[0], 1]))

            prepare_time = time.time() - start_time
            z = encoder.forward(x, input_pos)
            x_out = decoder.forward(z, prop_pos, input_pos)

            pred_loss = rel_loss(x_out, y, 2)

            gt_deriv = central_diff(y, dx)
            pred_deriv = central_diff(x_out, dx)
            deriv_loss = rel_loss(pred_deriv, gt_deriv, 2)

            loss = pred_loss + 1e-3*deriv_loss
            enc_optim.zero_grad()
            dec_optim.zero_grad()

            loss.backward()
            # with amp.scale_loss(loss, [enc_optim, dec_optim]) as scaled_loss:
            #     scaled_loss.backward()
            # print(torch.max(decoder.decoding_transformer.attn_module1.to_q.weight.grad))
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

            # Unscales gradients and calls
            enc_optim.step()
            dec_optim.step()

            enc_scheduler.step()
            dec_scheduler.step()

            # udpate tensorboardX
            writer.add_scalar('train_loss', loss, n_iter)
            writer.add_scalar('prediction_loss', pred_loss, n_iter)

            # compute computation time [None,...]and *compute_efficiency*
            process_time = time.time() - start_time - prepare_time

            pbar.set_description(
                f'Total loss (1e-4): {loss.item()*1e4:.1f}||'
                f'prediction (1e-4): {pred_loss.item()*1e4:.1f}||'
                f'derivative (1e-4): {deriv_loss.item()*1e4:.1f}||'
                f'Iters: {n_iter}/{opt.iters}')

            pbar.update(1)
            start_time = time.time()
            n_iter += 1

            if (n_iter-1) % opt.ckpt_every == 0 or n_iter >= opt.iters:
                logger.info('Tesing')
                print('Testing')

                encoder.eval()
                decoder.eval()

                with torch.no_grad():
                    all_avg_loss = []
                    all_acc_loss = []
                    visualization_cache = {
                        'in_seq': [],
                        'pred': [],
                        'gt': [],
                    }
                    picked = 0
                    for j, data in enumerate(tqdm(test_dataloader)):
                        # data preparation
                        x, y = data

                        if use_cuda:
                            x, y = x.cuda(), y.cuda()

                        # standardize
                        # data_mean = torch.mean(x, dim=1, keepdim=True)
                        # data_std = torch.std(x, dim=1, keepdim=True)
                        # x = (x - data_mean) / data_std
                        # y = (y - data_mean) / data_std
                        data_mean = 0.
                        data_std = 1.

                        input_pos = prop_pos = gridx.repeat([x.shape[0], 1, 1])
                        #x, input_pos = pad_pbc(x, input_pos, dx)
                        x = torch.cat((x, input_pos), dim=-1)  # concat coordinates as an additional feature

                        prepare_time = time.time() - start_time
                        z = encoder.forward(x, input_pos)
                        x_out = decoder.forward(z, prop_pos, input_pos)

                        avg_loss = rel_loss(x_out, y, p=2)
                        accumulated_mse = torch.nn.MSELoss(reduction='sum')(x_out*data_std, y*data_std) /   \
                                          (res**2 * x.shape[0])

                        all_avg_loss += [avg_loss.item()]
                        all_acc_loss += [accumulated_mse.item()]

                        # rescale
                        x = x[:, :, :1] * data_std + data_mean
                        x_out = x_out * data_std + data_mean
                        y = y * data_std + data_mean

                        if picked < 64:
                            idx = np.arange(0, min(64 - picked, x.shape[0]))
                            # randomly pick a batch
                            x = x[idx]
                            y = y[idx]
                            x_out = x_out[idx]
                            visualization_cache['gt'].append(y)
                            visualization_cache['in_seq'].append(x)
                            visualization_cache['pred'].append(x_out)
                            picked += x.shape[0]

                all_gt = torch.cat(visualization_cache['gt'], dim=0)
                all_in_seq = torch.cat(visualization_cache['in_seq'], dim=0)
                all_pred = torch.cat(visualization_cache['pred'], dim=0)

                make_image_grid(all_in_seq, all_pred, all_gt,
                                os.path.join(sample_dir, f'result_iter:{n_iter}_{j}.png'))

                del visualization_cache
                writer.add_scalar('testing avg loss', np.mean(all_avg_loss), global_step=n_iter)

                print(f'Testing avg loss (1e-4): {np.mean(all_avg_loss)*1e4}')
                print(f'Testing accumulated mse loss (1e-4): {np.mean(all_acc_loss)*1e4}')

                logger.info(f'Current iteration: {n_iter}')
                logger.info(f'Testing avg loss (1e-4): {np.mean(all_avg_loss)*1e4}')
                logger.info(f'Testing accumulated mse loss (1e-4): {np.mean(all_acc_loss)*1e4}')

                # save checkpoint if needed
                ckpt = {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'n_iter': n_iter,
                    'enc_optim': enc_optim.state_dict(),
                    'dec_optim': dec_optim.state_dict(),
                    'enc_sched': enc_scheduler.state_dict(),
                    'dec_sched': dec_scheduler.state_dict(),
                }

                save_checkpoint(ckpt, os.path.join(checkpoint_dir, f'model_checkpoint{n_iter}.ckpt'))
                del ckpt
                if n_iter >= opt.iters:
                    break

