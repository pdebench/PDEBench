## 'FNO'
# Advection
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta0.4.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta1.0.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta0.4.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta1.0.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='FNO' ++args.if_training=False
# Reaction Diffusion
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu0.5_Rho1.0.hdf5' ++args.model_name='FNO' ++args.reduced_resolution_t=1
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu0.5_Rho10.0.hdf5' ++args.model_name='FNO' ++args.reduced_resolution_t=1
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu2.0_Rho1.0.hdf5' ++args.model_name='FNO' ++args.reduced_resolution_t=1
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu2.0_Rho10.0.hdf5' ++args.model_name='FNO' ++args.reduced_resolution_t=1
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu0.5_Rho1.0.hdf5' ++args.model_name='FNO' ++args.reduced_resolution_t=1 ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu0.5_Rho10.0.hdf5' ++args.model_name='FNO' ++args.reduced_resolution_t=1 ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu2.0_Rho1.0.hdf5' ++args.model_name='FNO' ++args.reduced_resolution_t=1 ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu2.0_Rho10.0.hdf5' ++args.model_name='FNO' ++args.reduced_resolution_t=1 ++args.if_training=False
# Burgers Eq.
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.001.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.01.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.1.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='FNO'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.001.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.01.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.1.hdf5' ++args.model_name='FNO' ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='FNO' ++args.if_training=False
## Unet
# Advection
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta0.4.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta1.0.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta0.4.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta1.0.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Advection_Sols_beta4.0.hdf5' ++args.model_name='Unet' ++args.if_training=False
# Reaction Diffusion
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu0.5_Rho1.0.hdf5' ++args.model_name='Unet' ++args.reduced_resolution_t=1
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu0.5_Rho10.0.hdf5' ++args.model_name='Unet' ++args.reduced_resolution_t=1
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu2.0_Rho1.0.hdf5' ++args.model_name='Unet' ++args.reduced_resolution_t=1
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu2.0_Rho10.0.hdf5' ++args.model_name='Unet' ++args.reduced_resolution_t=1
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu0.5_Rho1.0.hdf5' ++args.model_name='Unet' ++args.reduced_resolution_t=1 ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu0.5_Rho10.0.hdf5' ++args.model_name='Unet' ++args.reduced_resolution_t=1 ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu2.0_Rho1.0.hdf5' ++args.model_name='Unet' ++args.reduced_resolution_t=1 ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='ReacDiff_Nu2.0_Rho10.0.hdf5' ++args.model_name='Unet' ++args.reduced_resolution_t=1 ++args.if_training=False
# Burgers Eq.
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.001.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.01.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.1.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='Unet'
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.001.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.01.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu0.1.hdf5' ++args.model_name='Unet' ++args.if_training=False
CUDA_VISIBLE_DEVICES='2' python3 train_models_forward.py +args=config.yaml ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='Unet' ++args.if_training=False
