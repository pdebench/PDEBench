## 'FNO'
# Advection
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta0.1.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta0.4.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta1.0.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml  ++args.filename='1D_Advection_Sols_beta0.1.hdf5' ++args.model_name='FNO'++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta0.4.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta1.0.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='FNO' ++args.if_training=False
# Reaction Diffusion
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu0.5_Rho1.0.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu0.5_Rho10.0.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu2.0_Rho1.0.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu2.0_Rho10.0.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu0.5_Rho1.0.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu0.5_Rho10.0.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu2.0_Rho1.0.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu2.0_Rho10.0.hdf5' ++args.model_name='FNO' ++args.if_training=False
# Burgers Eq.
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.001.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.01.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.1.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.001.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.01.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.1.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='FNO' ++args.if_training=False
## Unet
# Advection
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta0.1.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta0.4.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta1.0.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta0.1.hdf5' ++args.model_name='Unet'++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta0.4.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta1.0.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Adv.yaml ++args.filename='1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='Unet' ++args.if_training=False
# Reaction Diffusion
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu0.5_Rho1.0.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu0.5_Rho10.0.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu2.0_Rho1.0.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu2.0_Rho10.0.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu0.5_Rho1.0.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu0.5_Rho10.0.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu2.0_Rho1.0.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_ReacDiff.yaml ++args.filename='ReacDiff_Nu2.0_Rho10.0.hdf5' ++args.model_name='Unet' ++args.if_training=False
# Burgers Eq.
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.001.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.01.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.1.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.001.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.01.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu0.1.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='Unet' ++args.if_training=False
# 'PINN'
# Advection
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='1D_Advection_Sols_beta0.1.hdf5' ++args.aux_params=[0.1]
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='1D_Advection_Sols_beta0.4.hdf5' ++args.aux_params=[0.4]
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='1D_Advection_Sols_beta1.0.hdf5' ++args.aux_params=[1.]
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='1D_Advection_Sols_beta4.0.hdf5' ++args.aux_params=[4.]
# Reaction Diffusion
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='ReacDiff_Nu0.5_Rho1.0.hdf5' ++args.aux_params=[0.5,1.] ++args.val_time=0.5
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='ReacDiff_Nu0.5_Rho10.0.hdf5' ++args.aux_params=[0.5,10.] ++args.val_time=0.5
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='ReacDiff_Nu2.0_Rho1.0.hdf5' ++args.aux_params=[2.,1.] ++args.val_time=0.5
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='ReacDiff_Nu2.0_Rho10.0.hdf5' ++args.aux_params=[2.,10.] ++args.val_time=0.5
# Burgers Eq.	
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='1D_Burgers_Sols_Nu0.001.hdf5' ++args.aux_params=[0.001]
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='1D_Burgers_Sols_Nu0.01.hdf5' ++args.aux_params=[0.01]
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='1D_Burgers_Sols_Nu0.1.hdf5' ++args.aux_params=[0.1]
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_pinn_pde1d.yaml ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' ++args.aux_params=[1.]
