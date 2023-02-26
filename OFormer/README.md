Implementation of [Transformer for PDE Operator Learning](Transformer for Partial Differential Equations’ Operator Learning), a.k.a. [Operator Transformer (OFormer)](https://arxiv.org/abs/2205.13671).



### Training

* Train on Navier-Stokes dataset:

```bash
python tune_navier_stokes.py \
--lr 5e-4 \
--ckpt_every 10000 \
--iters 128000 \
--batch_size 16 \
--in_seq_len 10 \
--out_seq_len 20 \
--dataset_path ../pde_data/fno_ns_Re200_N10000_T30.npy \   # path to the dataset
--in_channels 12 \
--out_channels 1 \
--encoder_emb_dim 96 \
--out_seq_emb_dim 192 \
--encoder_depth 5 \
--decoder_emb_dim 384 \
--propagator_depth 1 \
--out_step 1 \
--train_seq_num 9800 \
--test_seq_num 200 \
--fourier_frequency 8 \
--encoder_heads 1 \
--use_grad \
--curriculum_ratio 0.16 \
--curriculum_steps 10 \
--aug_ratio 0.0
```

* Train on **1D Burgers'** PDE: 

```bash
python train_burgers.py \
--ckpt_every 1000 \
--iters 20000 \
--lr 8e-4 \
--batch_size 16 \
--dataset_path ../pde_data/burgers_data_R10.mat \   # path to dataset
--train_seq_num 1024 \
--test_seq_num 100 \
--resolution 2048
```

* Train on Darcy flow:

```bash
python train_darcy.py \
--ckpt_every 2000 \
--iters 32000 \
--lr 8e-4 \
--batch_size 8 \
--train_dataset_path ../../../pde_data/Darcy_421/piececonst_r421_N1024_smooth1.mat \
--test_dataset_path ../../../pde_data/Darcy_421/piececonst_r421_N1024_smooth2.mat \
--train_seq_num 1024 \
--test_seq_num 100 \
--resolution 141
```
