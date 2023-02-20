import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce, partial

# import data loaders, task-specific losses and metrics
from data_loaders import load_imagenet, load_text, load_cifar, load_mnist, load_deepsea, load_darcy_flow, load_psicov, load_ecg, load_satellite, load_ninapro, load_cosmic, load_spherical, load_fsd, load_domainnet, load_pde, load_openml, load_drug
from utils import FocalLoss, LpLoss, conv_init, get_params_to_update, set_param_grad, set_grad_state
from utils import mask, accuracy, accuracy_onehot, auroc, psicov_mae, ecg_f1, fnr, map_value, inv_auroc, r2_score, inverse_score, auc_metric, nmse, rmse_loss, nrmse_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(root, dataset, batch_size, valid_split, maxsize=None, get_shape=False):
    data_kwargs = None

    if dataset == "your_new_task": # modify this to experiment with a new task
        train_loader, val_loader, test_loader = None, None, None
    elif dataset == "DOMAINNET":
        train_loader, val_loader, test_loader = load_domainnet(root, batch_size, valid_split=valid_split)
    elif dataset == "IMAGENET":
        train_loader, val_loader, test_loader = load_imagenet(root, batch_size, maxsize=maxsize)
    elif dataset == "text":
        train_loader, val_loader, test_loader = load_text(root, batch_size, maxsize=maxsize)
    elif dataset == "CIFAR10":
        train_loader, val_loader, test_loader = load_cifar(root, 10, batch_size, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "CIFAR10-PERM":
        train_loader, val_loader, test_loader = load_cifar(root, 10, batch_size, permute=True, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "CIFAR100":
        train_loader, val_loader, test_loader = load_cifar(root, 100, batch_size, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "CIFAR100-PERM":
        train_loader, val_loader, test_loader = load_cifar(root, 100, batch_size, permute=True, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "MNIST":
        train_loader, val_loader, test_loader = load_mnist(root, batch_size, valid_split=valid_split)
    elif dataset == "MNIST-PERM":
        train_loader, val_loader, test_loader = load_mnist(root, batch_size, permute=True, valid_split=valid_split)
    elif dataset == "SPHERICAL":
        train_loader, val_loader, test_loader = load_spherical(root, batch_size, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "DEEPSEA":
        train_loader, val_loader, test_loader = load_deepsea(root, batch_size, valid_split=valid_split)
    elif dataset == "DARCY-FLOW-5":
        train_loader, val_loader, test_loader, y_normalizer = load_darcy_flow(root, batch_size, sub = 5, valid_split=valid_split)
        data_kwargs = {"decoder": y_normalizer}
    elif dataset == 'PSICOV':
        train_loader, val_loader, test_loader, _, _ = load_psicov(root, batch_size, valid_split=valid_split)
    elif dataset == "ECG":
        train_loader, val_loader, test_loader = load_ecg(root, batch_size, valid_split=valid_split)
    elif dataset == "SATELLITE":
        train_loader, val_loader, test_loader = load_satellite(root, batch_size, valid_split=valid_split)
    elif dataset == "NINAPRO":
        train_loader, val_loader, test_loader = load_ninapro(root, batch_size, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "COSMIC":
        train_loader, val_loader, test_loader = load_cosmic(root, batch_size, valid_split=valid_split)
        data_kwargs = {'transform': mask}
    elif dataset == "FSD":
        train_loader, val_loader, test_loader = load_fsd(root, batch_size, valid_split=valid_split)
    elif dataset[:3] == 'PDE':
        train_loader, val_loader, test_loader = load_pde(root, batch_size, dataset=dataset[4:], valid_split=valid_split)
    elif dataset[:6] == 'OPENML':
        train_loader, val_loader, test_loader = load_openml(root, batch_size, int(dataset[6:]), valid_split=valid_split, get_shape=get_shape)
    elif dataset[:4] == 'DRUG':
        train_loader, val_loader, test_loader = load_drug(root, batch_size, dataset[5:], valid_split=valid_split)

    n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)

    if not valid_split:
        val_loader = test_loader
        n_val = n_test

    return train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs


def get_config(root, args):
    dataset = args.dataset
    args.infer_label = False
    args.activation = None
    args.target_seq_len = 512 if not hasattr(args, 'target_seq_len') else args.target_seq_len
    print("target_seq_len", args.target_seq_len)
    
    if dataset == "your_new_task": # modify this to experiment with a new task
        dims, num_classes = None, None
        loss = None

    elif dataset == "DOMAINNET":
        dims, sample_shape, num_classes = 1, (1, 3, 224, 224), 40
        loss = nn.CrossEntropyLoss()

    elif dataset[:5] == "CIFAR":
        dims, sample_shape, num_classes = 2,  (1, 3, 32, 32), 10 if dataset in ['CIFAR10', 'CIFAR10-PERM'] else 100
        loss = nn.CrossEntropyLoss()

    elif dataset == 'SPHERICAL':
        dims, sample_shape, num_classes = 2, (1, 3, 60, 60), 100
        loss = nn.CrossEntropyLoss() 

    elif dataset == "DARCY-FLOW-5":
        dims, sample_shape, num_classes = 2, (1, 3, 85, 85), 1
        loss = LpLoss(size_average=False)
        args.infer_label = True

    elif dataset == "PSICOV":
        dims, sample_shape, num_classes = 2, (1, 57, 512, 512), 1
        loss = nn.MSELoss(reduction='mean')
        args.infer_label = True

    elif dataset == "NINAPRO": 
        dims, sample_shape, num_classes = 2, (1, 1, 16, 52), 18
        loss = FocalLoss(alpha=1)

    elif dataset == "COSMIC":
        dims, sample_shape, num_classes = 2, (1, 1, 128, 128), 1
        loss = nn.BCEWithLogitsLoss()
        args.infer_label = True

    elif dataset == 'FSD':
        dims, sample_shape, num_classes = 2, (1, 1, 96, 102), 200
        loss = nn.BCEWithLogitsLoss(pos_weight=10 * torch.ones((200, )))
        args.infer_label = True
        
    elif dataset[:5] == "MNIST":
        dims, sample_shape, num_classes = 1, (1, 1, 784), 10
        loss = F.nll_loss
    
    elif dataset == "ECG": 
        dims, sample_shape, num_classes = 1, (1, 1, 1000), 4
        loss = nn.CrossEntropyLoss()   

    elif dataset == "SATELLITE":
        dims, sample_shape, num_classes = 1, (1, 1, 46), 24
        loss = nn.CrossEntropyLoss()

    elif dataset == "DEEPSEA":
        dims, sample_shape, num_classes = 1, (1, 4, 1000), 36
        loss = nn.BCEWithLogitsLoss(pos_weight=4 * torch.ones((36, )))
        args.infer_label = True

    elif dataset == 'PDE-Burgers':
        dims, sample_shape, num_classes = 1, (1, 1, 256), (1, 1024)
        loss = rmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-1DCFD':
        dims, sample_shape, num_classes = 1, (1, 1, 3072), (1, 3072) 
        loss = rmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-ADV':
        dims, sample_shape, num_classes = 1, (1, 1, 256), (1, 256)
        loss = nrmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-RD':
        dims, sample_shape, num_classes = 1, (1, 1, 1024), (1, 1024) 
        loss = nrmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-DS':
        dims, sample_shape, num_classes = 1, (1, 1, 1024), (1, 1024)
        loss = nrmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-SW':
        dims, sample_shape, num_classes = 1, (1, 1, 128, 128), 1
        loss = nrmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-RD2D':
        dims, sample_shape, num_classes = 1, (1, 1, 128, 128), 1
        loss = nrmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-Darcy':
        dims, sample_shape, num_classes = 1, (1, 1, 64, 64), 1
        loss = nrmse_loss 
        args.infer_label = True

    elif dataset == 'PDE-2DCFD':
        dims, sample_shape, num_classes = 1, (1, 4, 64, 64), (1, 4, 64, 64)
        loss = nrmse_loss 
        args.infer_label = True
    
    elif dataset[:6] == 'OPENML':
        train_loader, val_loader, test_loader = load_openml(root, 1, int(dataset[6:]), get_shape=True)
        sample_shape = (1, train_loader.dataset.tensors[0].size(dim=-2), train_loader.dataset.tensors[0].size(dim=-1))
        num_classes = int(train_loader.dataset.tensors[1].max().item() + 1)
        dims = 1
        weights = []
        for c in range(num_classes):
            weights.append(1.0 / ((train_loader.dataset.tensors[1]==c).float().mean().item()))

        loss = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        print("OPENML dataset id:", int(dataset[6:]), " sample shape:", sample_shape, " num classes: ", num_classes, "loss: ", loss, "weights:", weights)

    elif dataset[:4] == 'DRUG':
        dims, sample_shape, num_classes = 1, (1, 1, 3840), 1
        loss = nn.MSELoss(reduction='mean')
        args.infer_label = True

    return dims, sample_shape, num_classes, loss, args


def get_metric(root, dataset):
    if dataset == "your_new_task": # modify this to experiment with a new task
        return inverse_score(accuracy), np.min
    if dataset[:5] == "CIFAR" or dataset[:5] == "MNIST" or dataset == 'NINAPRO' or dataset == "SATELLITE" or dataset == "SPHERICAL" or dataset == "DOMAINNET":
        return inverse_score(accuracy), np.min
    if dataset == "DEEPSEA":
        return inverse_score(auroc), np.min
    if dataset == "DARCY-FLOW-5":
        return LpLoss(size_average=True), np.min
    if dataset[:3] == 'PDE':
        return nmse, np.min
    if dataset == 'PSICOV':
        return psicov_mae(root), np.min
    if dataset == 'ECG':
        return inverse_score(ecg_f1), np.min
    if dataset == 'COSMIC':
        return inv_auroc, np.min
    if dataset == 'FSD':
        return inverse_score(map_value), np.min
    if dataset[:4] == 'DRUG':
        return inverse_score(r2_score), np.min
    if dataset[:6] == 'OPENML':
        return inverse_score(auc_metric), np.min
        # return inverse_score(accuracy), np.min


def get_optimizer(name, params):
    if name == 'SGD':
        return partial(torch.optim.SGD, lr=params['lr'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    elif name == 'Adam':
        return partial(torch.optim.Adam, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])
    elif name == 'AdamW':
        return partial(torch.optim.AdamW, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])


def get_scheduler(name, params, epochs=200, n_train=None):
    if name == 'StepLR':
        sched = params['sched']

        def scheduler(epoch):    
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break
                    
            return math.pow(params['base'], optim_factor)  

        lr_sched_iter = False

    elif name == 'WarmupLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return f  

    elif name == 'ExpLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return params['base'] * f  

    elif name == 'SinLR':

        cycles = 0.5
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            # progress after warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1. + math.cos(math.pi * float(cycles) * 2.0 * progress)))

    return scheduler, lr_sched_iter


def get_optimizer_scheduler(args, model, module=None, n_train=1):
    if module is None:
        set_grad_state(model, True)
        set_param_grad(model, args.finetune_method)
        optimizer = get_optimizer(args.optimizer.name, args.optimizer.params)(get_params_to_update(model, ""))
        lr_lambda, args.lr_sched_iter = get_scheduler(args.scheduler.name, args.scheduler.params, args.epochs, n_train)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return args, model, optimizer, scheduler

    elif module == 'embedder':
        embedder_optimizer_params = copy.deepcopy(args.optimizer.params)
        if embedder_optimizer_params['lr'] <= 0.001:
            embedder_optimizer_params['lr'] *= 10
        embedder_optimizer = get_optimizer(args.optimizer.name, embedder_optimizer_params)(get_params_to_update(model, ""))
        lr_lambda, _ = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.embedder_epochs, 1)
        embedder_scheduler = torch.optim.lr_scheduler.LambdaLR(embedder_optimizer, lr_lambda=lr_lambda)

        return args, model, embedder_optimizer, embedder_scheduler

    elif module == 'predictor':

        try:
            predictor = model.predictor
            set_grad_state(model, False)
            for n, m in model.embedder.named_parameters():
                m.requires_grad = True
            for n, m in model.predictor.named_parameters():
                m.requires_grad = True

            predictor_optimizer_params = copy.deepcopy(args.optimizer.params)
            if predictor_optimizer_params['lr'] <= 0.001:
                predictor_optimizer_params['lr'] *= 10
            predictor_optimizer = get_optimizer(args.optimizer.name, predictor_optimizer_params)(get_params_to_update(model, ""))
            lr_lambda, args.lr_sched_iter = get_scheduler(args.no_warmup_scheduler.name, args.no_warmup_scheduler.params, args.predictor_epochs, 1)
            predictor_scheduler = torch.optim.lr_scheduler.LambdaLR(predictor_optimizer, lr_lambda=lr_lambda)

            return args, model, predictor_optimizer, predictor_scheduler
        except:
            print("No predictor module.")

