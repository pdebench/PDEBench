#!/bin/bash
nn=1
key=2020
while [ "$nn" -le 10 ]; do
  CUDA_VISIBLE_DEVICES='0,1' python3 CFD_multi_Hydra.py +args=1D_Multi_trans.yaml ++args.init_key="$key"
  nn=$(${nn} + 1)
  key=$(${key} + 1)
  echo "$nn"
  echo "$key"
done
