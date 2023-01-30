nn=1
key=2020
while [ $nn -le 50 ]; do
  CUDA_VISIBLE_DEVICES='0,1' python3 reaction_diffusion_2D_multi_solution_Hydra.py +multi=config_2D.yaml ++multi.init_k\
ey=$key
  nn=$(expr $nn + 1)
  key=$(expr $key + 1)
  echo "$nn"
  echo "$key"
done