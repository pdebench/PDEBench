# /bin/bash
# F.Alesiani, 2022, June 6th

# Train forward model
HYDRA_FULL_ERROR=1 python3 train_models_inverse.py ++args.filename='/1D/Advection/Train/1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='FNO'
HYDRA_FULL_ERROR=1 python3 train_models_inverse.py ++args.filename='/1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='FNO'
HYDRA_FULL_ERROR=1 python3 train_models_inverse.py ++args.filename='/1D/ReactionDiffusion/Train/ReacDiff_Nu1.0_Rho2.0.hdf5' ++args.model_name='FNO'
HYDRA_FULL_ERROR=1 python3 train_models_inverse.py ++args.filename='/1D/CFD/Train/1D_CFD_Shock_trans_Train.hdf5' ++args.model_name='FNO' ++args.in_channels=3 ++args.out_channels=3 ++args.num_channels=3 ++args.final_time=5  

HYDRA_FULL_ERROR=1 python3 train_models_inverse.py ++args.filename='/1D/Advection/Train/1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='Unet'
HYDRA_FULL_ERROR=1 python3 train_models_inverse.py ++args.filename='/1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='Unet'
HYDRA_FULL_ERROR=1 python3 train_models_inverse.py ++args.filename='/1D/ReactionDiffusion/Train/ReacDiff_Nu1.0_Rho2.0.hdf5' ++args.model_name='Unet'
HYDRA_FULL_ERROR=1 python3 train_models_inverse.py ++args.filename='/1D/CFD/Train/1D_CFD_Shock_trans_Train.hdf5' ++args.model_name='Unet' ++args.in_channels=3 ++args.out_channels=3 ++args.num_channels=3 ++args.final_time=5  


# Inverse
HYDRA_FULL_ERROR=1 python3 inverse/train.py ++args.filename='/1D/Advection/Train/1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='FNO'
HYDRA_FULL_ERROR=1 python3 inverse/train.py ++args.filename='/1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='FNO'
HYDRA_FULL_ERROR=1 python3 inverse/train.py ++args.filename='/1D/ReactionDiffusion/Train/ReacDiff_Nu1.0_Rho2.0.hdf5' ++args.model_name='FNO'
HYDRA_FULL_ERROR=1 python3 inverse/train.py ++args.filename='/1D/CFD/Train/1D_CFD_Shock_trans_Train.hdf5' ++args.model_name='FNO' ++args.in_channels=3 ++args.out_channels=3 ++args.num_channels=3 ++args.final_time=5  

HYDRA_FULL_ERROR=1 python3 inverse/train.py ++args.filename='/1D/Advection/Train/1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='Unet'
HYDRA_FULL_ERROR=1 python3 inverse/train.py ++args.filename='/1D/Burgers/Train/1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='Unet'
HYDRA_FULL_ERROR=1 python3 inverse/train.py ++args.filename='/1D/ReactionDiffusion/Train/ReacDiff_Nu1.0_Rho2.0.hdf5' ++args.model_name='Unet'
HYDRA_FULL_ERROR=1 python3 inverse/train.py ++args.filename='/1D/CFD/Train/1D_CFD_Shock_trans_Train.hdf5' ++args.model_name='Unet' ++args.in_channels=3 ++args.out_channels=3 ++args.num_channels=3 ++args.final_time=5  


