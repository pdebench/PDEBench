# -*- coding: utf-8 -*-
"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     utils.py
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


import matplotlib.pyplot as plt

def plot_ic_solution_mcmc(latent,x,y,grid,model_inverse,model,device,fname_save="IC_inverse_problem_mcmc.pdf"):
    """
    Plots the prediction of the initial condition estimated using MCMC from the latent with the model "model"
    y  = model(x)
    y[i] = model(latent[i]), i =0, ... 

    June 2022, F.Alesiani
    """        
    fig, axes = plt.subplots(1,2,figsize=(15,7))
    ax  = axes[0]
    u0 = model_inverse.latent2source(latent[0]).to(device)
    pred_u0 = model(u0, grid)
    ax.plot(u0.detach().cpu().flatten(),'r',label="Predicted Initial Condition")
    for _latent in latent:
        u0 = model_inverse.latent2source(_latent).to(device)
        ax.plot(u0.detach().cpu().flatten(),'r',alpha=0.1)
    ax.plot(x.detach().cpu().flatten(),'b--',label="True Initial Condition")
    ax.legend()
    # plt.show()

    ax  = axes[1]
    ax.plot(pred_u0.detach().cpu().flatten(),'r',label="Predicted forward value")
    ax.plot(y.detach().cpu().flatten(),'b--',label="True forward value")
    for _latent in latent:
        u0 = model_inverse.latent2source(_latent).to(device)
        pred_u0 = model(u0, grid)
        ax.plot(pred_u0.detach().cpu().flatten(),'r',alpha=0.1)
    ax.legend()
    if fname_save:
        plt.savefig(fname_save, bbox_inches='tight')



def plot_ic_solution_grad(model_ic,x,y,grid,model,device,fname_save="IC_inverse_problem_grad.pdf"):
    """
    Plots the prediction of the initial condition estimated using model_ic with the model "model"
    y  = model(x)
    y' = model(model_ic())

    June 2022, F.Alesiani
    """    

    fig, axes = plt.subplots(1,2,figsize=(15,7))
    ax  = axes[0]
    u0 = model_ic().to(device).unsqueeze(0).unsqueeze(-1)
    pred_u0 = model(u0, grid)
    ax.plot(u0.detach().cpu().flatten(),'r',label="Predicted Initial Condition")
    ax.plot(x.detach().cpu().flatten(),'b--',label="True Initial Condition")
    ax.legend()
    # plt.show()

    ax  = axes[1]
    ax.plot(pred_u0.detach().cpu().flatten(),'r',label="Predicted forward value")
    ax.plot(y.detach().cpu().flatten(),'b--',label="True forward value")
    ax.legend()
    if fname_save:
        plt.savefig(fname_save, bbox_inches='tight')        


from scipy.signal import welch
import matplotlib.pyplot as plt

def plot_ic_solution_grad_psd(model_ic,x,y,grid,model,device,fname_save="IC_inverse_problem_grad_psd.pdf"):
    """
    Plots the prediction of the initial condition estimated using model_ic with the model "model"
    y  = model(x)
    y' = model(model_ic())
    It also shows the power density

    June 2022, F.Alesiani
    """    
    fig, axes = plt.subplots(1,3,figsize=(22,7))
    ax  = axes[0]
    u0 = model_ic().to(device).unsqueeze(0).unsqueeze(-1)
    pred_u0 = model(u0, grid)
    ax.plot(u0.detach().cpu().flatten(),'r',label="Predicted Initial Condition")
    ax.plot(x.detach().cpu().flatten(),'b--',label="True Initial Condition")
    ax.legend()
    # plt.show()

    ax  = axes[1]
    ax.plot(pred_u0.detach().cpu().flatten(),'r',label="Predicted forward value")
    ax.plot(y.detach().cpu().flatten(),'b--',label="True forward value")
    ax.legend()

    _u0 = u0.detach().cpu().flatten()
    _x = x[0].detach().cpu().flatten()

    fz = u0.shape[1]

    fu,puu = welch(_u0,fz)
    fx,pxx = welch(_x,fz)

    ax  = axes[2]
    ax.semilogy(fu,puu,'r',label="predicted u0")
    ax.semilogy(fx,pxx,'b--',label="x true")
    ax.set_xlabel('spatial frequency')
    ax.set_ylabel('PSD')
    ax.legend()

    if fname_save:
        plt.savefig(fname_save, bbox_inches='tight')           



import sys, os
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
import pandas as pd
import numpy as np


def get_metric_name(filename,model_name, base_path,inverse_model_type):
    """
    returns the name convention for the result file

    June 2022, F.Alesiani
    """
    inverse_metric_filename = base_path + filename[:-5] + '_' + model_name +'_'+ inverse_model_type + ".pickle"    
    return inverse_metric_filename

def read_results(model_names,inverse_model_type, base_path, filenames,shortfilenames, verbose=False):
    """
    reads and merges the result files. 
    Shortnames are used for the name of the dataset as alternative to the file name.

    June 2022, F.Alesiani
    """
    dfs = []
    for model_name in model_names:
        for filename,shortfilename in zip(filenames,shortfilenames):
            # print(filename)
            inverse_metric_filename = get_metric_name(filename,model_name, base_path,inverse_model_type)
            if verbose: print ("reading resul file: ",inverse_metric_filename)
            df = pd.read_pickle(inverse_metric_filename)
            df['model'] = model_name
            df['pde'] = shortfilename
            dfs+=[df]
    keys = ['pde','model']
    df = pd.concat(dfs,axis=0)
    return df, keys

@hydra.main(config_path='../config', config_name='results')
def process_results(cfg: DictConfig):
    """
    reads and merges the result files and aggregate the results with the selected values. The results are aggregated by datafile. 

    June 2022, F.Alesiani
    """    
    print(cfg.args)
  
    df, keys = read_results(cfg.args.model_names,cfg.args.inverse_model_type, cfg.args.base_path, cfg.args.filenames, cfg.args.shortfilenames)
    df1p3 = df[keys + list(cfg.args.results_values)]
    df2p3 = df1p3.groupby(by=keys).agg([np.mean,np.std]).reset_index()
    print("saving results into: ", cfg.args.base_path + cfg.args.result_filename)
    df2p3.to_csv(cfg.args.base_path + cfg.args.result_filename)            


if __name__ == "__main__":
    process_results()
    print("Done.")    