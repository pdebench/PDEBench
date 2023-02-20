## Cross-Modal Fine-Tuning: Align then Refine

Original PyTorch implementation of ORCA proposed in the paper "[Cross-Modal Fine-Tuning: Align then Refine](https://arxiv.org/abs/)". 
ORCA is developed for effectively solving  ML problems in diverse modalities using large-scale pretrained transformers. 
It adapts to a target task via an align-then-refine workflow: given the target input, ORCA first learns an embedding network that aligns the embedded feature distribution with the pretraining modality. The pretrained model is then fine-tuned on the embedded data to exploit the knowledge shared across modalities. 

This adaptation of the original repo specifically supports:
- transferring [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta) and [Swin Transformers](https://huggingface.co/docs/transformers/model_doc/swin) (Hugging Face implementation) to downstream tasks;
- minimizing the L2 distance, Maximum Mean Descrepancy ([MMD](https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_pytorch.py)), or Optimal Transport Dataset Distance ([OTDD](https://github.com/microsoft/otdd)) for distributional alignment;
- replicate experiments on [PDEBench](https://github.com/pdebench/PDEBench) simulation task.

## Installation

Create a Mamba environment:

- `mamba env create --file orca_pdebench.mml`
- `bash src/startup-hook.sh` to install the dependencies.

## Experiments on PDEBench dataset

1. Download the required datasets (see `src/datasets`)
2. Download the precomputed language features [text_xs.py](https://www.dropbox.com/s/yhlf25n8rzmdrtp/text_xs.npy?dl=0) and [text_ys.py](https://www.dropbox.com/s/16lj1vprg1pzckt/text_ys.npy?dl=0) (if you are using [RoBERTa models](https://huggingface.co/docs/transformers/model_doc/roberta)) to  `src/datasets`
3. Train and evaluate a certain PDE
```
python src/main.py --config src/configs/PDEx.yaml
```




## Citations
To cite the original ORCA paper, use:
```bibtex
@inproceedings{shen2023orca,
  title={Cross-Modal Fine-Tuning: Align then Refine},
    year={2023}
}
```
To cite PDEBench datasets, use:

```bibtex
@data{darus-2986_2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
publisher = {DaRUS},
title = {{PDEBench Datasets}},
year = {2022},
doi = {10.18419/darus-2986},
url = {https://doi.org/10.18419/darus-2986}
}
```

To cite the PDEBench benchmark paper, use:

```bibtex
@inproceedings{PDEBench2022,
author = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Dan and Alesiani, Francesco and Pflüger, Dirk and Niepert, Mathias},
title = {{PDEBench: An Extensive Benchmark for Scientific Machine Learning}},
year = {2022},
booktitle = {36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks},
url = {https://arxiv.org/abs/2210.07182}
}
```
