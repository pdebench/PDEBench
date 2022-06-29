"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     Data_Merge.py
  Authors:  Makoto Takamoto (makoto.takamoto@neclab.eu)

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
'''
Data_Merge.py
This is a script creating HDF5 from the generated data (numpy array) by our data generation scripts.
A more detailed explanation how to use this script is provided in the README. 
'''


import numpy as np
import h5py
import glob

# Hydra
import hydra
from omegaconf import DictConfig

def _mergeRD(var, DataND, savedir):
    _vars = ['2D', 'nu']
    if var not in _vars:
        print(var+' is not defined!')
        return None

    idx = 0
    datas = glob.glob(savedir+'/' + var + '*key*.npy')
    datas.sort()
    for data in datas:
        print(idx, data)
        test = np.load(data).squeeze()
        batch = min(test.shape[0], DataND.shape[0] - idx)
        if var == '2D':
            DataND[idx:idx + batch] = test[:batch, -2]
        else:
            DataND[idx:idx + batch] = test[:batch]
        idx += batch

    return DataND[:idx]

def _merge(var, DataND, dim, savedir):
    if dim == 1:
        _vars = ['D', 'P', 'Vx']
    elif dim == 2:
        _vars = ['D', 'P', 'Vx', 'Vy']
    elif dim == 3:
        _vars = ['D', 'P', 'Vx', 'Vy', 'Vz']
    if var not in _vars:
        print(var+' is not defined!')
        return None

    idx = 0
    datas = glob.glob(savedir+'/HD*' + var + '.npy')
    datas.sort()
    for data in datas:
        print(idx, data)
        test = np.load(data).squeeze()
        batch = min(test.shape[0], DataND.shape[0] - idx)
        DataND[idx:idx+batch] = test[:batch]
        idx += batch

    return DataND[:idx]

def nan_check(data):
    data = np.abs(data).reshape([data.shape[0], data.shape[1],-1]).sum(axis=-1)
    return np.where(data[:,-2] < 1.e-6)[0], np.where(data[:,-2] > 1.e-6)[0]

def merge(type, dim, bd, nbatch, savedir):
    if type=='CFD':
        datas = glob.glob(savedir+'/HD*D.npy')
        datas.sort()
        test = np.load(datas[0])
        __nbatch, nt, nx, ny, nz = test.shape
        _nbatch = __nbatch * len(datas)
        print('nb, nt, nx, ny, nz: ', _nbatch, nt, nx, ny, nz)
        print('nbatch: {0}, _nbatch: {1}'.format(nbatch, _nbatch))
        assert nbatch <= _nbatch, 'nbatch should be equal or less than the number of generated samples'
        assert 2*nbatch > _nbatch, '2*nbatch should be larger than the number of generated samples'

        if dim == 1:
            DataND = np.zeros([2*nbatch, nt, nx], dtype=np.float32)
            vars = ['D', 'P', 'Vx']
        elif dim == 2:
            DataND = np.zeros([2*nbatch, nt, nx, ny], dtype=np.float32)
            vars = ['D', 'P', 'Vx', 'Vy']
        elif dim == 3:
            DataND = np.zeros([2*nbatch, nt, nx, ny, nz], dtype=np.float32)
            vars = ['D', 'P', 'Vx', 'Vy', 'Vz']

    elif type=='ReacDiff':
        datas = glob.glob(savedir+'/nu*.npy')
        datas.sort()
        test = np.load(datas[0])
        __nbatch, nx, ny = test.shape
        _nbatch = __nbatch * len(datas)
        print('nbatch: {0}, _nbatch: {1}'.format(nbatch, _nbatch))
        assert nbatch == _nbatch, 'nbatch should be equal or less than the number of generated samples'
        print('nb, nx, ny: ', _nbatch, nx, ny)
        DataND = np.zeros([nbatch, nx, ny], dtype=np.float32)
        vars = ['2D', 'nu']

    for var in vars:
        if type=='CFD':
            _DataND = _merge(var, DataND, dim, savedir)
            if var=='D':
                idx_neg, idx_pos = nan_check(_DataND)
                print('idx_neg: {0}, idx_pos: {1}'.format(len(idx_neg), len(idx_pos)))
                if len(idx_pos) < nbatch:
                    print('too many ill-defined data...')
                    print('nbatch: {0}, idx_pos: {1}'.format(nbatch, len(idx_pos)))
            _DataND = _DataND[idx_pos]
            _DataND = _DataND[:nbatch]
            np.save(savedir+'/' + var + '.npy', _DataND)
        elif type == 'ReacDiff':
            DataND = _mergeRD(var, DataND, savedir)
            np.save(savedir+'/' + var + '.npy', DataND)

    datas = glob.glob(savedir+'/*npy')
    datas.sort()

    if type == 'CFD':
        zcrd = np.load(datas[-1])
        del (datas[-1])
    ycrd = np.load(datas[-1])
    del (datas[-1])
    xcrd = np.load(datas[-1])
    del (datas[-1])
    tcrd = np.load(datas[-1])
    del (datas[-1])
    if type=='ReacDiff':
        #datas = glob.glob('save/' + type + '/nu*key*npy')
        datas = glob.glob(savedir+'/nu*key*npy')
        datas.sort()
        _beta = datas[0].split('/')[-1].split('_')[3]
        flnm = savedir+'/2D_DecayFlow_' + _beta + '_Train.hdf5'
        with h5py.File(flnm, 'w') as f:
            f.create_dataset('tensor', data=np.load(savedir+'/2D.npy')[:, None, :, :])
            f.create_dataset('nu', data=np.load(savedir+'/nu.npy'))
            f.create_dataset('x-coordinate', data=xcrd)
            f.create_dataset('y-coordinate', data=ycrd)
            f.attrs['beta'] = float(_beta[4:])
        return 0

    mode = datas[1].split('/')[-1].split('_')[3]
    _eta = datas[1].split('/')[-1].split('_')[4]
    _zeta = datas[1].split('/')[-1].split('_')[5]
    _M = datas[1].split('/')[-1].split('_')[6]
    if dim == 1:
        flnm = savedir+'/1D_CFD_' + mode + '_' + _eta + '_' + _zeta + '_' + bd + '_Train.hdf5'
    elif dim == 2:
        flnm = savedir+'/2D_CFD_' + mode + '_' + _eta + '_' + _zeta + '_' + _M + '_' + bd + '_Train.hdf5'
    elif dim == 3:
        flnm = savedir+'/3D_CFD_' + mode + '_' + _eta + '_' + _zeta + '_' + _M + '_' + bd + '_Train.hdf5'
    print(flnm)

    del(DataND)

    with h5py.File(flnm, 'w') as f:
        f.create_dataset('density', data=np.load(savedir+'/D.npy'))
        f.create_dataset('pressure', data=np.load(savedir+'/P.npy'))
        f.create_dataset('Vx', data=np.load(savedir+'/Vx.npy'))
        if dim > 1:
            f.create_dataset('Vy', data=np.load(savedir+'/Vy.npy'))
            f.create_dataset('y-coordinate', data=ycrd)
        if dim == 3:
            f.create_dataset('Vz', data=np.load(savedir+'/Vz.npy'))
            f.create_dataset('z-coordinate', data=zcrd)
        f.create_dataset('x-coordinate', data = xcrd)
        f.create_dataset('t-coordinate', data = tcrd)
        eta = float(_eta[3:])
        zeta = float(_zeta[4:])
        print('(eta, zeta) = ', eta, zeta)
        f.attrs['eta'] = eta
        f.attrs['zeta'] = zeta
        if dim > 1:
            M = float(_M[1:])
            f.attrs['M'] = M
            print('M: ', M)

def transform(type, savedir):
    datas = glob.glob(savedir+'/*npy')
    datas.sort()
    xcrd = np.load(datas[-1])
    del (datas[-1])
    tcrd = np.load(datas[-1])
    del (datas[-1])

    flnm = datas[0]
    with h5py.File(flnm[:-3]+'hdf5', 'w') as f:
        print(flnm)
        _data = np.load(flnm)
        f.create_dataset('tensor', data = _data.astype(np.float32))
        f.create_dataset('x-coordinate', data = xcrd)
        f.create_dataset('t-coordinate', data = tcrd)
        if type=='advection':
            beta = float(flnm.split('/')[-1].split('_')[3][4:-4])  # advection train
            print(beta)
            f.attrs['beta'] = beta

        elif type=='burgers':
            Nu = float(flnm.split('/')[-1].split('_')[-1][2:-4])  # Burgers test/train
            print(Nu)
            f.attrs['Nu'] = Nu

        elif type=='ReacDiff':
            Rho = float(flnm.split('/')[-1].split('_')[-1][3:-4])  # reac-diff test
            Nu = float(flnm.split('/')[-1].split('_')[-2][2:])  # reac-diff test
            print(Nu, Rho)
            f.attrs['Nu'] = Nu
            f.attrs['rho'] = Rho

# Init arguments with Hydra
@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    pde1ds = ['advection', 'burgers', 'ReacDiff']
    if cfg.args.type in pde1ds and cfg.args.dim==1:
        transform(type=cfg.args.type, savedir=cfg.args.savedir)
    else:
        bds = ['periodic', 'trans']
        assert cfg.args.bd in bds, 'bd should be either periodic or trans'
        merge(type=cfg.args.type, dim=cfg.args.dim, bd=cfg.args.bd, nbatch=cfg.args.nbatch, savedir=cfg.args.savedir)

if __name__=='__main__':
    main()