# -*- coding: utf-8 -*-
"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     metrics.py
  Authors:  Makoto Takamoto (makoto.takamoto@neclab.eu)
            Francesco Alesiani (francesco.alesiani@neclab.eu)

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

import torch
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def metric_func(pred, target, if_mean=True, Lx=1., Ly=1., Lz=1., iLow=4, iHigh=12):
    """
    code for calculate metrics discussed in the Brain-storming session
    RMSE, normalized RMSE, max error, RMSE at the boundaries, conserved variables, RMSE in Fourier space, temporal sensitivity
    """
    pred, target = pred.to(device), target.to(device)
    # (batch, nx^i..., timesteps, nc)
    idxs = target.size()
    if len(idxs) == 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    if len(idxs) == 5:
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)
    idxs = target.size()
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]

    # RMSE
    err_mean = torch.sqrt(torch.mean((pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])) ** 2, dim=2))
    err_RMSE = torch.mean(err_mean, axis=0)
    nrm = torch.sqrt(torch.mean(target.view([nb, nc, -1, nt]) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)

    err_CSV = torch.sqrt(torch.mean(
        (torch.sum(pred.view([nb, nc, -1, nt]), dim=2) - torch.sum(target.view([nb, nc, -1, nt]), dim=2)) ** 2,
        dim=0))
    if len(idxs) == 4:
        nx = idxs[2]
        err_CSV /= nx
    elif len(idxs) == 5:
        nx, ny = idxs[2:4]
        err_CSV /= nx * ny
    elif len(idxs) == 6:
        nx, ny, nz = idxs[2:5]
        err_CSV /= nx * ny * nz
    # worst case in all the data
    err_Max = torch.max(torch.max(
        torch.abs(pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])), dim=2)[0], dim=0)[0]

    if len(idxs) == 4:  # 1D
        err_BD = (pred[:, :, 0, :] - target[:, :, 0, :]) ** 2
        err_BD += (pred[:, :, -1, :] - target[:, :, -1, :]) ** 2
        err_BD = torch.mean(torch.sqrt(err_BD / 2.), dim=0)
    elif len(idxs) == 5:  # 2D
        nx, ny = idxs[2:4]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD = (torch.sum(err_BD_x, dim=-2) + torch.sum(err_BD_y, dim=-2)) / (2 * nx + 2 * ny)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)
    elif len(idxs) == 6:  # 3D
        nx, ny, nz = idxs[2:5]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD_z = (pred[:, :, :, :, 0] - target[:, :, :, :, 0]) ** 2
        err_BD_z += (pred[:, :, :, :, -1] - target[:, :, :, :, -1]) ** 2
        err_BD = torch.sum(err_BD_x.view([nb, -1, nt]), dim=-2) \
                 + torch.sum(err_BD_y.view([nb, -1, nt]), dim=-2) \
                 + torch.sum(err_BD_z.view([nb, -1, nt]), dim=-2)
        err_BD = err_BD / (2 * nx * ny + 2 * ny * nz + 2 * nz * nx)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)

    if len(idxs) == 4:  # 1D
        nx = idxs[2]
        pred_F = torch.fft.rfft(pred, dim=2)
        target_F = torch.fft.rfft(target, dim=2)
        _err_F = torch.sqrt(torch.mean(torch.abs(pred_F - target_F) ** 2, axis=0)) / nx * Lx
    if len(idxs) == 5:  # 2D
        pred_F = torch.fft.fftn(pred, dim=[2, 3])
        target_F = torch.fft.fftn(target, dim=[2, 3])
        nx, ny = idxs[2:4]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt]).to(device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                it = mt.floor(mt.sqrt(i ** 2 + j ** 2))
                if it > min(nx // 2, ny // 2) - 1:
                    continue
                err_F[:, :, it] += _err_F[:, :, i, j]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny) * Lx * Ly
    elif len(idxs) == 6:  # 3D
        pred_F = torch.fft.fftn(pred, dim=[2, 3, 4])
        target_F = torch.fft.fftn(target, dim=[2, 3, 4])
        nx, ny, nz = idxs[2:5]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2, nz // 2), nt]).to(device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                for k in range(nz // 2):
                    it = mt.floor(mt.sqrt(i ** 2 + j ** 2 + k ** 2))
                    if it > min(nx // 2, ny // 2, nz // 2) - 1:
                        continue
                    err_F[:, :, it] += _err_F[:, :, i, j, k]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny * nz) * Lx * Ly * Lz

    err_F = torch.zeros([nc, 3, nt]).to(device)
    err_F[:,0] += torch.mean(_err_F[:,:iLow], dim=1)  # low freq
    err_F[:,1] += torch.mean(_err_F[:,iLow:iHigh], dim=1)  # middle freq
    err_F[:,2] += torch.mean(_err_F[:,iHigh:], dim=1)  # high freq

    if if_mean:
        return torch.mean(err_RMSE, dim=[0, -1]), \
               torch.mean(err_nRMSE, dim=[0, -1]), \
               torch.mean(err_CSV, dim=[0, -1]), \
               torch.mean(err_Max, dim=[0, -1]), \
               torch.mean(err_BD, dim=[0, -1]), \
               torch.mean(err_F, dim=[0, -1])
    else:
        return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F

def metrics(val_loader, model, Lx, Ly, Lz, plot, channel_plot, model_name, x_min,
            x_max, y_min, y_max, t_min, t_max, mode='FNO', initial_step=None, ):
    if mode=='Unet':
        with torch.no_grad():
            itot = 0
            for xx, yy in val_loader:
                xx = xx.to(device)
                yy = yy.to(device)

                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                for t in range(initial_step, yy.shape[-2]):
                    inp = xx.reshape(inp_shape)
                    temp_shape = [0, -1]
                    temp_shape.extend([i for i in range(1,len(inp.shape)-1)])
                    inp = inp.permute(temp_shape)
                    
                    y = yy[..., t:t+1, :]
                
                    temp_shape = [0]
                    temp_shape.extend([i for i in range(2,len(inp.shape))])
                    temp_shape.append(1)
                    im = model(inp).permute(temp_shape).unsqueeze(-2)
                    pred = torch.cat((pred, im), -2)
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
                    = metric_func(pred, yy, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)

                if itot == 0:
                    err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F \
                        = _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F
                    pred_plot = pred[:1]
                    target_plot = yy[:1]
                    val_l2_time = torch.zeros(yy.shape[-2]).to(device)
                else:
                    err_RMSE += _err_RMSE
                    err_nRMSE += _err_nRMSE
                    err_CSV += _err_CSV
                    err_Max += _err_Max
                    err_BD += _err_BD
                    err_F += _err_F
                    
                    mean_dim = [i for i in range(len(yy.shape)-2)]
                    mean_dim.append(-1)
                    mean_dim = tuple(mean_dim)
                    val_l2_time += torch.sqrt(torch.mean((pred-yy)**2, dim=mean_dim))
                
                itot += 1

    elif mode=='FNO':
        with torch.no_grad():
            itot = 0
            for xx, yy, grid in val_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                grid = grid.to(device)

                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                for t in range(initial_step, yy.shape[-2]):
                    inp = xx.reshape(inp_shape)
                    y = yy[..., t:t + 1, :]
                    im = model(inp, grid)
                    pred = torch.cat((pred, im), -2)
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
                    = metric_func(pred, yy, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)
                if itot == 0:
                    err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F \
                        = _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F
                    pred_plot = pred[:1]
                    target_plot = yy[:1]
                    val_l2_time = torch.zeros(yy.shape[-2]).to(device)
                else:
                    err_RMSE += _err_RMSE
                    err_nRMSE += _err_nRMSE
                    err_CSV += _err_CSV
                    err_Max += _err_Max
                    err_BD += _err_BD
                    err_F += _err_F
                    
                    mean_dim = [i for i in range(len(yy.shape)-2)]
                    mean_dim.append(-1)
                    mean_dim = tuple(mean_dim)
                    val_l2_time += torch.sqrt(torch.mean((pred-yy)**2, dim=mean_dim))

                itot += 1

    elif mode == "PINN":
        raise NotImplementedError


    err_RMSE = np.array(err_RMSE.data.cpu()/itot)
    err_nRMSE = np.array(err_nRMSE.data.cpu()/itot)
    err_CSV = np.array(err_CSV.data.cpu()/itot)
    err_Max = np.array(err_Max.data.cpu()/itot)
    err_BD = np.array(err_BD.data.cpu()/itot)
    err_F = np.array(err_F.data.cpu()/itot)
    print('RMSE: {0:.5f}'.format(err_RMSE))
    print('normalized RMSE: {0:.5f}'.format(err_nRMSE))
    print('RMSE of conserved variables: {0:.5f}'.format(err_CSV))
    print('Maximum value of rms error: {0:.5f}'.format(err_Max))
    print('RMSE at boundaries: {0:.5f}'.format(err_BD))
    print('RMSE in Fourier space: {0}'.format(err_F))
    
    val_l2_time = val_l2_time/itot
    
    if plot:
        dim = len(yy.shape) - 3
        plt.ioff()
        if dim == 1:
            
            fig, ax = plt.subplots(figsize=(6.5,6))
            h = ax.imshow(pred_plot[...,channel_plot].squeeze().detach().cpu(),
                       extent=[t_min, t_max, x_min, x_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,channel_plot].min(), target_plot[...,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Prediction", fontsize=30)
            ax.tick_params(axis='x',labelsize=30)
            ax.tick_params(axis='y',labelsize=30)
            ax.set_ylabel("$x$", fontsize=30)
            ax.set_xlabel("$t$", fontsize=30)
            plt.tight_layout()
            filename = model_name + '_pred.pdf'
            plt.savefig(filename)
            
            fig, ax = plt.subplots(figsize=(6.5,6))
            h = ax.imshow(target_plot[...,channel_plot].squeeze().detach().cpu(),
                       extent=[t_min, t_max, x_min, x_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,channel_plot].min(), target_plot[...,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Data", fontsize=30)
            ax.tick_params(axis='x',labelsize=30)
            ax.tick_params(axis='y',labelsize=30)
            ax.set_ylabel("$x$", fontsize=30)
            ax.set_xlabel("$t$", fontsize=30)
            plt.tight_layout()
            filename = model_name + '_data.pdf'
            plt.savefig(filename)
    
        elif dim == 2:
            
            fig, ax = plt.subplots(figsize=(6.5,6))
            h = ax.imshow(pred_plot[...,-1,channel_plot].squeeze().t().detach().cpu(),
                       extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,-1,channel_plot].min(), target_plot[...,-1,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Prediction", fontsize=30)
            ax.tick_params(axis='x',labelsize=30)
            ax.tick_params(axis='y',labelsize=30)
            ax.set_ylabel("$y$", fontsize=30)
            ax.set_xlabel("$x$", fontsize=30)
            plt.tight_layout()
            filename = model_name + '_pred.pdf'
            plt.savefig(filename)
            
            fig, ax = plt.subplots(figsize=(6.5,6))
            h = ax.imshow(target_plot[...,-1,channel_plot].squeeze().t().detach().cpu(),
                       extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            h.set_clim(target_plot[...,-1,channel_plot].min(), target_plot[...,-1,channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Data", fontsize=30)
            ax.tick_params(axis='x',labelsize=30)
            ax.tick_params(axis='y',labelsize=30)
            ax.set_ylabel("$y$", fontsize=30)
            ax.set_xlabel("$x$", fontsize=30)
            plt.tight_layout()
            filename = model_name + '_data.pdf'
            plt.savefig(filename)
    
        # plt.figure(figsize=(8,8))
        # plt.semilogy(torch.arange(initial_step,yy.shape[-2]),
        #              val_l2_time[initial_step:].detach().cpu())
        # plt.xlabel('$t$', fontsize=30)
        # plt.ylabel('$MSE$', fontsize=30)
        # plt.title('MSE vs unrolled time steps', fontsize=30)
        # plt.tight_layout()
        # filename = model_name + '_mse_time.pdf'
        # plt.savefig(filename)
        
        filename = model_name + 'mse_time.npz'
        np.savez(filename, t=torch.arange(initial_step,yy.shape[-2]).cpu(),
                 mse=val_l2_time[initial_step:].detach().cpu())

    return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F


# LpLoss Function  
class LpLoss(object):
    """
    Lp loss function 
    """
    def __init__(self, p=2, reduction='mean'):
        super(LpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert p > 0
        self.p = p
        self.reduction = reduction
    def __call__(self, x, y, eps=1e-20):
        num_examples = x.size()[0]
        _diff = x.view(num_examples,-1) - y.view(num_examples,-1)
        _diff = torch.norm(_diff, self.p, 1)
        _norm = eps + torch.norm(y.view(num_examples,-1), self.p, 1)
        if self.reduction in ['mean']:
            return torch.mean(_diff/_norm)
        if self.reduction in ['sum']:
            return torch.sum(_diff/_norm)
        return _diff/_norm

# FftLoss Function  
class FftLpLoss(object):
    """
    loss function in Fourier space

    June 2022, F.Alesiani
    """
    def __init__(self, p=2, reduction='mean'):
        super(FftLpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert p > 0
        self.p = p
        self.reduction = reduction
    def __call__(self, x, y, flow=None,fhigh=None, eps=1e-20):
        num_examples = x.size()[0]
        others_dims = x.shape[1:]
        dims = list(range(1,len(x.shape)))
        xf = torch.fft.fftn(x,dim=dims)
        yf = torch.fft.fftn(y,dim=dims)
        if flow is None: flow = 0
        if fhigh is None: fhigh = np.max(xf.shape[1:])

        if len(others_dims) ==1:
            xf = xf[:,flow:fhigh]
            yf = yf[:,flow:fhigh]        
        if len(others_dims) ==2:
            xf = xf[:,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh]
        if len(others_dims) ==3:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh]
        if len(others_dims) ==4:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]

        _diff = xf - yf.reshape(xf.shape)
        _diff = torch.norm(_diff.reshape(num_examples,-1), self.p, 1)
        _norm = eps + torch.norm(yf.reshape(num_examples,-1), self.p, 1)
        
        if self.reduction in ['mean']:
            return torch.mean(_diff/_norm)
        if self.reduction in ['sum']:
            return torch.sum(_diff/_norm)
        return _diff/_norm

import torch.nn.functional as F
# FftLoss Function  
class FftMseLoss(object):
    """
    loss function in Fourier space

    June 2022, F.Alesiani
    """
    def __init__(self, reduction='mean'):
        super(FftMseLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        self.reduction = reduction
    def __call__(self, x, y, flow=None,fhigh=None, eps=1e-20):
        num_examples = x.size()[0]
        others_dims = x.shape[1:-2]
        for d in others_dims:
            assert (d>1), "we expect the dimension to be the same and greater the 1"
        # print(others_dims)
        dims = list(range(1,len(x.shape)-1))
        xf = torch.fft.fftn(x,dim=dims)
        yf = torch.fft.fftn(y,dim=dims)
        if flow is None: flow = 0
        if fhigh is None: fhigh = np.max(xf.shape[1:])

        if len(others_dims) ==1:
            xf = xf[:,flow:fhigh]
            yf = yf[:,flow:fhigh]        
        if len(others_dims) ==2:
            xf = xf[:,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh]
        if len(others_dims) ==3:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh]
        if len(others_dims) ==4:
            xf = xf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]
            yf = yf[:,flow:fhigh,flow:fhigh,flow:fhigh,flow:fhigh]
        _diff = xf - yf
        _diff = _diff.reshape(num_examples,-1).abs()**2
        if self.reduction in ['mean']:
            return torch.mean(_diff).abs()
        if self.reduction in ['sum']:
            return torch.sum(_diff).abs()
        return _diff.abs()


import torch.nn as nn
def inverse_metrics(u0,x,pred_u0,y):
    """
    computes all the metrics in the base and fourier space
    u0: esimated initial condition, 
    pred_u0: prediction from the esimated initial condition, pred_u0 = model(u0)
    x: true initial condition
    y: true prediction, y = model(x)

    June 2022, F.Alesiani
    """
    
    mseloss_fn = nn.MSELoss(reduction="mean")
    l2loss_fn = LpLoss(p=2,reduction='mean')
    l3loss_fn = LpLoss(p=3,reduction='mean')
    
    fftmseloss_fn = FftMseLoss(reduction="mean")
    fftl2loss_fn = FftLpLoss(p=2,reduction="mean")
    fftl3loss_fn = FftLpLoss(p=3,reduction="mean")    

    #initial condition
    mseloss_u0 = mseloss_fn(u0.view(1, -1), x.view(1, -1)).item()
    l2loss_u0 = l2loss_fn(u0.view(1, -1), x.view(1, -1)).item()
    l3loss_u0 = l3loss_fn(u0.view(1, -1), x.view(1, -1)).item()
    
    
    fmid = u0.shape[1]//4

    fftmseloss_u0 = fftmseloss_fn(u0, x).item()
    fftmseloss_low_u0 = fftmseloss_fn(u0, x, 0, fmid).item()
    fftmseloss_mid_u0 = fftmseloss_fn(u0, x, fmid, 2*fmid).item()
    fftmseloss_hi_u0 = fftmseloss_fn(u0, x, 2*fmid).item()
    
    fftl2loss_u0 = fftl2loss_fn(u0, x).item()
    fftl2loss_low_u0 = fftl2loss_fn(u0, x, 0, fmid).item()
    fftl2loss_mid_u0 = fftl2loss_fn(u0, x, fmid, 2*fmid).item()
    fftl2loss_hi_u0 = fftl2loss_fn(u0, x, 2*fmid).item()

    fftl3loss_u0 = fftl3loss_fn(u0, x).item()
    fftl3loss_low_u0 = fftl3loss_fn(u0, x, 0, fmid).item()
    fftl3loss_mid_u0 = fftl3loss_fn(u0, x, fmid, 2*fmid).item()
    fftl3loss_hi_u0 = fftl3loss_fn(u0, x, 2*fmid).item()    

    #prediction 
    mseloss_pred_u0 = mseloss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()
    l2loss_pred_u0 = l2loss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()
    l3loss_pred_u0 = l3loss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()

    fmid = pred_u0.shape[1]//4
    pred_u0 = pred_u0.squeeze(-1)
    y = y.squeeze(-1)
    
    fftmseloss_pred_u0 = fftmseloss_fn(pred_u0, y).item()
    fftmseloss_low_pred_u0 = fftmseloss_fn(pred_u0, y, 0, fmid).item()
    fftmseloss_mid_pred_u0 = fftmseloss_fn(pred_u0, y, fmid, 2*fmid).item()
    fftmseloss_hi_pred_u0 = fftmseloss_fn(pred_u0, y, 2*fmid).item()
    
    fftl2loss_pred_u0 = fftl2loss_fn(pred_u0, y).item()
    fftl2loss_low_pred_u0 = fftl2loss_fn(pred_u0, y, 0, fmid).item()
    fftl2loss_mid_pred_u0 = fftl2loss_fn(pred_u0, y, fmid, 2*fmid).item()
    fftl2loss_hi_pred_u0= fftl2loss_fn(pred_u0, y, 2*fmid).item()

    fftl3loss_pred_u0 = fftl3loss_fn(pred_u0, y).item()
    fftl3loss_low_pred_u0 = fftl3loss_fn(pred_u0, y, 0, fmid).item()
    fftl3loss_mid_pred_u0 = fftl3loss_fn(pred_u0, y, fmid, 2*fmid).item()
    fftl3loss_hi_pred_u0 = fftl3loss_fn(pred_u0, y, 2*fmid).item()   

    metric = {
        'mseloss_u0': mseloss_u0
        ,'l2loss_u0': l2loss_u0
        ,'l3loss_u0': l3loss_u0

        ,'mseloss_pred_u0': mseloss_pred_u0
        ,'l2loss_pred_u0': l2loss_pred_u0
        ,'l3loss_pred_u0': l3loss_pred_u0


        ,'fftmseloss_u0': fftmseloss_u0
        ,'fftmseloss_low_u0': fftmseloss_low_u0
        ,'fftmseloss_mid_u0': fftmseloss_mid_u0
        ,'fftmseloss_hi_u0': fftmseloss_hi_u0



        ,'fftmseloss_pred_u0': fftmseloss_pred_u0
        ,'fftmseloss_low_pred_u0': fftmseloss_low_pred_u0
        ,'fftmseloss_mid_pred_u0': fftmseloss_mid_pred_u0
        ,'fftmseloss_hi_pred_u0': fftmseloss_hi_pred_u0

        ,'fftl2loss_u0': fftl2loss_u0
        ,'fftl2loss_low_u0': fftl2loss_low_u0
        ,'fftl2loss_mid_u0': fftl2loss_mid_u0
        ,'fftl2loss_hi_u0': fftl2loss_hi_u0

        ,'fftl2loss_pred_u0': fftl2loss_pred_u0
        ,'fftl2loss_low_pred_u0': fftl2loss_low_pred_u0
        ,'fftl2loss_mid_pred_u0': fftl2loss_mid_pred_u0
        ,'fftl2loss_hi_pred_u0': fftl2loss_hi_pred_u0

        ,'fftl3loss_u0': fftl3loss_u0
        ,'fftl3loss_low_u0': fftl3loss_low_u0
        ,'fftl3loss_mid_u0': fftl3loss_mid_u0
        ,'fftl3loss_hi_u0': fftl3loss_hi_u0

        ,'fftl3loss_pred_u0': fftl3loss_pred_u0
        ,'fftl3loss_low_pred_u0': fftl3loss_low_pred_u0
        ,'fftl3loss_mid_pred_u0': fftl3loss_mid_pred_u0
        ,'fftl3loss_hi_pred_u0': fftl3loss_hi_pred_u0        
    }    

    return metric
