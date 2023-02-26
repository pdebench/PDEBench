import os
import shutil
import torch


def save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = None):
    """Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
    """
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint.txt')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False):
    """
    Loads the torch model from a checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeeded!' % ckpt_path)
    return ckpt


def ensure_dir(dir_name: str):
    """
    Creates folder if one doesn't not exist.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_arguments(parser):
    # basic training settings
    parser.add_argument(
        '--lr', type=float, default=3e-4, help='Specifies learing rate for optimizer. (default: 1e-3)'
    )
    parser.add_argument(
        '--resume', action='store_true', help='If set resumes training from provided checkpoint. (default: None)'
    )
    parser.add_argument(
        '--path_to_resume', type=str, default='', help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--iters', type=int, default=100000, help='Number of training iterations. (default: 100k)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./', help='Path to log, save checkpoints. '
    )
    parser.add_argument(
        '--ckpt_every', type=int, default=5000, help='Save model checkpoints every x iterations. (default: 5k)'
    )

    # ===================================
    # for dataset
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Size of each batch (default: 16)'
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True, help='Path to dataset.'
    )

    parser.add_argument(
        '--train_seq_num', type=int, default=1000, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--test_seq_num', type=int, default=100, help='How many sequences in the training dataset.'
    )
    parser.add_argument(
        '--resolution', type=int, default=2048, help='The interval of when sample snapshots from sequence'
    )

    return parser