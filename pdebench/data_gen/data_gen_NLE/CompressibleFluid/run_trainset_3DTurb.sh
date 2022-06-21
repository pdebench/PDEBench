nn=1
key=2031
while [ $nn -le 6 ]; do
  CUDA_VISIBLE_DEVICES='0,1,2,3' python3 CFD_multi_Hydra.py +args=3D_Multi_TurbM1.yaml ++args.init_key=$key
  nn=$(expr $nn + 1)
  key=$(expr $key + 1)
  echo "$nn"
  echo "$key"
done
