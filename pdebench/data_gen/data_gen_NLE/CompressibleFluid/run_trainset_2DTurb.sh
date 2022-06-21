nn=1
key=2031
#while [ $nn -le 100 ]; do
while [ $nn -le 55 ]; do
  CUDA_VISIBLE_DEVICES='0,1,2,3' python3 CFD_multi_Hydra.py +args=2D_Multi_Turb.yaml ++args.init_key=$key
  nn=$(expr $nn + 1)
  key=$(expr $key + 1)
  echo "$nn"
  echo "$key"
done
