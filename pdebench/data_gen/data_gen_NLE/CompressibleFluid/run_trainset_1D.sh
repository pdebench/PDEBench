nn=1
key=2020
while [ $nn -le 10 ]; do
  CUDA_VISIBLE_DEVICES='0,1' python3 CFD_multi_Hydra.py +args=1D_Multi.yaml ++args.init_key=$key
  nn=$(expr $nn + 1)
  key=$(expr $key + 1)
  echo "$nn"
  echo "$key"
done
