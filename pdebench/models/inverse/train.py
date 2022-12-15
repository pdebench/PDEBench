# -*- coding: utf-8 -*-
"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     inverse.py
  Authors:  Francesco Alesiani (makoto.takamoto@neclab.eu)
            Dan MacKinlay (Dan.MacKinlay@data61.csiro.au) 

NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

import sys
import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.infer import  MCMC, NUTS
from pyro import poutine

from timeit import default_timer


import sys, os
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict


import pdebench as pde
from pdebench.models.fno.fno import FNO1d,FNO2d,FNO3d
from pdebench.models.fno.utils import FNODatasetSingle, FNODatasetMult

from pdebench.models.unet.unet import UNet1d, UNet2d, UNet3d
from pdebench.models.unet.utils import UNetDatasetSingle,UNetDatasetMult

from pdebench.models import metrics
from pdebench.models.metrics import LpLoss,FftLpLoss,FftMseLoss,inverse_metrics
import pandas as pd


from pdebench.models.inverse.inverse import ProbRasterLatent, ElementStandardScaler, InitialConditionInterp
from pdebench.models.inverse.utils import plot_ic_solution_mcmc

from torch.distributions.normal import Normal 

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model,model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


@hydra.main(config_path='../config', config_name='config')
def main(cfg: DictConfig):
    print(cfg.args.filename)
    print(cfg.args)

     # we use the test data
    if cfg.args.model_name in ['FNO']:
        inverse_data = FNODatasetSingle(cfg.args.filename,
                                saved_folder = cfg.args.base_path,
                                reduced_resolution=cfg.args.reduced_resolution,
                                reduced_resolution_t=cfg.args.reduced_resolution_t,
                                reduced_batch=cfg.args.reduced_batch,
                                initial_step=cfg.args.initial_step,
                                if_test=True,
                                num_samples_max = cfg.args.num_samples_max
                                )

        _data, _, _ = next(iter(inverse_loader))
        dimensions = len(_data.shape)
        spatial_dim = dimensions - 3

    if cfg.args.model_name in ['UNET','Unet']:
        inverse_data = UNetDatasetSingle(cfg.args.filename,
                                saved_folder = cfg.args.base_path,
                                reduced_resolution=cfg.args.reduced_resolution,
                                reduced_resolution_t=cfg.args.reduced_resolution_t,
                                reduced_batch=cfg.args.reduced_batch,
                                initial_step=cfg.args.initial_step,
                                if_test=True,
                                num_samples_max = cfg.args.num_samples_max)                            

        inverse_loader = torch.utils.data.DataLoader(inverse_data, batch_size=1,shuffle=False)
        _data, _  = next(iter(inverse_loader))
        dimensions = len(_data.shape)
        spatial_dim = dimensions - 3

    initial_step = cfg.args.initial_step
    t_train = cfg.args.t_train
    
    model_name = cfg.args.filename[:-5] + '_' + cfg.args.model_name
    model_path = cfg.args.base_path + model_name + ".pt"

    if cfg.args.model_name in ['FNO']:
        if dimensions == 4:
            print(cfg.args.num_channels)
            model = FNO1d(num_channels=cfg.args.num_channels,
                            width=cfg.args.width,
                            modes=cfg.args.modes,
                            initial_step=cfg.args.initial_step).to(device)

        if dimensions == 5:
            model = FNO2d(num_channels=cfg.args.num_channels,
                            width=cfg.args.width,
                            modes1=cfg.args.modes,
                            modes2=cfg.args.modes,
                            initial_step=cfg.args.initial_step).to(device)                

        if dimensions == 6:
            model = FNO3d(num_channels=cfg.args.num_channels,
                            width=cfg.args.width,
                            modes1=cfg.args.modes,
                            modes2=cfg.args.modes,
                            modes3=cfg.args.modes,
                            initial_step=cfg.args.initial_step).to(device)                

    if cfg.args.model_name in ['UNET','Unet']:
        if dimensions == 4:
            model = UNet1d(cfg.args.in_channels, cfg.args.out_channels).to(device)
        elif dimensions == 5:
            model = UNet2d(cfg.args.in_channels, cfg.args.out_channels).to(device)
        elif dimensions == 6:
            model = UNet3d(cfg.args.in_channels, cfg.args.out_channels).to(device)    

    model = load_model(model,model_path, device)    

    model.eval()
    if cfg.args.inverse_model_type in ['ProbRasterLatent']:
        assert(spatial_dim==1), "give me time"
        if spatial_dim==1:
            ns,nx,nt,nc = _data.shape
            model_inverse = ProbRasterLatent(
                model.to(device),
                dims=[nx,1],
                latent_dims = [1,cfg.args.in_channels_hid,1],
                prior_scale = 0.1,
                obs_scale = 0.01,
                prior_std = 0.01,
                device=device
            )    

    if cfg.args.inverse_model_type in ['InitialConditionInterp']:
        loss_fn = nn.MSELoss(reduction="mean")
        input_dims = list(_data.shape[1:1+spatial_dim])        
        latent_dims = len(input_dims)*[cfg.args.in_channels_hid]
        if cfg.args.num_channels> 1:
            input_dims=input_dims+[cfg.args.num_channels]
            latent_dims=latent_dims+[cfg.args.num_channels]
        print(input_dims,latent_dims)
        model_ic = InitialConditionInterp(input_dims,latent_dims).to(device)
        model.to(device)


    scaler = ElementStandardScaler()
    loss_fn = nn.MSELoss(reduction="mean")

    inverse_u0_l2_full,inverse_y_l2_full = 0,0
    all_metric = []
    t1 = default_timer()
    for ks,sample in enumerate(inverse_loader):
        if cfg.args.model_name in ['FNO']:
            (xx, yy, grid) = sample
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)
            model_ = lambda x, grid: model(x,grid)

        if cfg.args.model_name in ['UNET','Unet']:
            (xx, yy) = sample
            grid = None
            xx = xx.to(device)
            yy = yy.to(device)
            model_ = lambda x, grid: model(x.permute([0, 2, 1])).permute([0, 2, 1])

        num_samples = ks + 1
        loss = 0


        x = xx[..., 0 , :]
        y = yy[..., t_train:t_train+1 , :]

        if ks==0:
            print(x.shape,y.shape)

        #scale the input and output
        x = scaler.fit_transform(x)
        y = scaler.transform(y)

        if cfg.args.inverse_model_type in ['ProbRasterLatent']:
            #Create model
            model_inverse.to(device)
            nuts_kernel = NUTS(model_inverse, full_mass=False, max_tree_depth=5, jit_compile=True) # high performacne config

            mcmc = MCMC(nuts_kernel, num_samples=cfg.args.mcmc_num_samples, warmup_steps=cfg.args.mcmc_warmup_steps, num_chains=cfg.args.mcmc_num_chains,disable_progbar=True)
            mcmc.run(grid, y)
            mc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}    

            # get the initial solution
            latent = torch.tensor(mc_samples['latent'])
            u0 = model_inverse.latent2source(latent[0]).to(device)
            pred_u0 = model(u0, grid)

        if cfg.args.inverse_model_type in ['InitialConditionInterp']:
            optimizer = torch.optim.Adam(model_ic.parameters(), lr=cfg.args.inverse_learning_rate, weight_decay=1e-4)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
            if cfg.args.inverse_verbose_flag:
                _iter = tqdm(range(cfg.args.inverse_epochs))
            else:
                _iter = range(cfg.args.inverse_epochs)
            for epoch in _iter:
                if cfg.args.num_channels>1:
                    u0 = model_ic().unsqueeze(0)
                else:
                    u0 = model_ic().unsqueeze(0).unsqueeze(-1)
                
                pred_u0 = model_(u0,grid)
                
                loss_u0 = loss_fn(pred_u0,y)
                optimizer.zero_grad()
                loss_u0.backward()
                optimizer.step()

                t2 = default_timer()
                if cfg.args.inverse_verbose_flag:
                    _iter.set_description(f"loss={loss_u0.item()}, t2-t1= {t2-t1}")        

        #compute losses            
        loss_u0 = loss_fn(u0.reshape(1, -1), x.reshape(1, -1)).item()
        loss_y = loss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()
        inverse_u0_l2_full += loss_u0
        inverse_y_l2_full += loss_y

        metric = inverse_metrics(u0,x,pred_u0,y)        
        metric['sample'] = ks

        all_metric+=[metric]            
        
        t2 = default_timer()
        print('samples: {}, loss_u0: {:.5f},loss_y: {:.5f}, t2-t1: {:.5f}, mse_inverse_u0_L2: {:.5f}, mse_inverse_y_L2: {:.5f}'\
            .format(ks+1, loss_u0, loss_y, t2 - t1, inverse_u0_l2_full/num_samples, inverse_y_l2_full/num_samples))

    df_metric = pd.DataFrame(all_metric)
    inverse_metric_filename = cfg.args.base_path + cfg.args.filename[:-5] + '_' + cfg.args.model_name +'_'+cfg.args.inverse_model_type + ".csv"    
    print("saving in :", inverse_metric_filename)
    df_metric.to_csv(inverse_metric_filename)

    inverse_metric_filename = cfg.args.base_path + cfg.args.filename[:-5] + '_' + cfg.args.model_name +'_'+cfg.args.inverse_model_type+ ".pickle"    
    print("saving in :", inverse_metric_filename)
    df_metric.to_pickle(inverse_metric_filename)

    inverse_metric_filename = cfg.args.base_path + cfg.args.filename[:-5] + '_' + cfg.args.model_name +'_'+cfg.args.inverse_model_type+ "_stats.csv"    
    print("saving in :", inverse_metric_filename)
    df_metric = df_metric.describe()
    df_metric.to_csv(inverse_metric_filename)

if __name__ == '__main__':
    main()
