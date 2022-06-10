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


class ElementStandardScaler:
  def fit(self, x):
    self.mean = x.mean()
    self.std = x.std(unbiased=False)
  def transform(self, x):
    eps = 1e-20 
    x = x - self.mean
    x = x/(self.std + eps)
    return x
  def fit_transform(self, x):
      self.fit(x)
      return self.transform(x)      

class ProbRasterLatent(PyroModule):
    def __init__(
            self,
            process_predictor: "nn.Module",
            dims = (256,256),
            latent_dims = (16,16),
            interpolation = "bilinear",
            prior_scale = 0.01,
            obs_scale = 0.01,
            prior_std = 0.01,
            device=None):
        super().__init__()
        self.dims = dims
        self.device = device
        self.prior_std = prior_std
        if latent_dims is None:
            latent_dims = dims
        self.latent_dims = latent_dims
        self.interpolation = interpolation
        self.prior_scale = prior_scale
        self.obs_scale = torch.tensor(obs_scale, device=self.device, dtype=torch.float)
        self.process_predictor = process_predictor
        process_predictor.train(False)
        ## Do not fit the process predictor weights
        for param in self.process_predictor.parameters():
            param.requires_grad = False
        _m,_s = torch.tensor([0], device=self.device, dtype=torch.float), torch.tensor([self.prior_std], device=self.device, dtype=torch.float) 
        self.latent = PyroSample(dist.Normal(_m,_s).expand(latent_dims).to_event(2))
        print(self.latent_dims,self.dims)

    def get_latent(self):
        if self.latent_dims==self.dims:
            return self.latent.unsqueeze(0)
        # `mini-batch x channels x [optional depth] x [optional height] x width`.
        l =  F.interpolate(
            self.latent.unsqueeze(1),
            self.dims,
            mode=self.interpolation,
            align_corners=False
        ).squeeze(0) #squeeze/unsqueeze is because of weird interpolate semantics
        return l

    def latent2source(self,latent):
        if latent.shape==self.dims:
            return latent.unsqueeze(0)
        # `mini-batch x channels x [optional depth] x [optional height] x width`.
        l =  F.interpolate(
            latent.unsqueeze(1),
            self.dims,
            mode=self.interpolation,
            align_corners=False
        ).squeeze(0) #squeeze/unsqueeze is because of weird interpolate semantics
        return l

    def forward(self, grid, y=None):
        #overwrite process predictor batch with my own latent
        x = self.get_latent()
        # print("forward:x.shape,grid.shape=",x.shape,grid.shape)
        mean = self.process_predictor(x.to(self.device),grid.to(self.device))
        o = pyro.sample(
            "obs", dist.Normal(mean, self.obs_scale).to_event(2),
            obs=y)
        return o    


import sys
import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial
from numpy import prod

class InitialConditionInterp(nn.Module):
    """
    InitialConditionInterp
    Class for the inital conditions using interpoliation. Works for 1d,2d and 3d

    model_ic = InitialConditionInterp([16],[8])
    model_ic = InitialConditionInterp([16,16],[8,8])
    model_ic = InitialConditionInterp([16,16,16],[8,8,8])

    June 2022, F.Alesiani
    """
    def __init__(self, dims, hidden_dim):
        super(InitialConditionInterp, self).__init__()
        self.spatial_dim = len(hidden_dim)
        self.dims = [1]+dims if len(dims)==1 else dims
        # self.dims = [1,1,1]+dims
        self.hidden_dim =  [1]+hidden_dim if len(hidden_dim)==1 else hidden_dim
        self.interpolation  = "bilinear" if len(hidden_dim)<3 else "trilinear"
        self.scale = (1 / prod(hidden_dim))
        self.latent = nn.Parameter(self.scale * torch.rand(1, 1, *self.hidden_dim, dtype=torch.float))
        # print(self.latent.shape)

    def latent2source(self,latent):
        if latent.shape[2:]==self.dims:
            return latent
        # `mini-batch x channels x [optional depth] x [optional height] x width`.
        l =  F.interpolate(
            latent,
            self.dims,
            mode=self.interpolation,
            align_corners=False
        ) 
        return l.view(self.dims)
    def forward(self):
        x = self.latent2source(self.latent)
        if self.spatial_dim == 1:
            x = x.squeeze(0)  
        return x