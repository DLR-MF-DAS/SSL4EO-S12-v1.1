[![arXiv](https://img.shields.io/badge/arXiv-2503.00168-b31b1b?logo=arxiv)](https://arxiv.org/abs/2503.00168)
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-embed2Scale-FFD21E?logo=huggingface)](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1)

# SSL4EO-S12 v1.1

SSL4EO-S12 v1.1 is an updated multimodal version of the popular EO pre-training dataset [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12).
Read more about the reasons behind our update and further improvements in our technical report on [arXiv](https://arxiv.org/abs/2503.00168).  

## NEWS

- Feb 17, 2026: SSL4EO-S12 v1.1 is now available as a webdataset version for better usability at [HuggingFace](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1).
- Mar 11, 2025: SSL4EO-S12 v1.1 available on [HuggingFace](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1-Zarr).
- Mar 10, 2025: SSL4EO-S12 v1.1 utilized as pre-training dataset for [2025 CVPR EARTHVISION data challenge](https://www.grss-ieee.org/events/earthvision-2025/?tab=challenge):
    * details: [https://github.com/DLR-MF-DAS/embed2scale-challenge-supplement](https://github.com/DLR-MF-DAS/embed2scale-challenge-supplement)
    * tech support: [https://github.com/DLR-MF-DAS/embed2scale-challenge-supplement/issues](https://github.com/DLR-MF-DAS/embed2scale-challenge-supplement/issues)


## Data

The dataset includes 246,144 locations with four timestamps each from the modalities S2L1C, S2L2A, S1GRD, S2RGB, NDVI, LULC, and a single timestamp DEM.
We refer to our [technical report](https://arxiv.org/abs/2503.00168) for details.

Sentinel-2 and Sentinel-1 time series examples with four seasonal images:

![ssl4eos12_timeseries.png](assets/ssl4eos12_timeseries.png)

Modality examples in SSL4EO-S12 v1.1:

![ssl4eoS12_modalities.png](assets/ssl4eoS12_modalities.png)

## Download

You can also download the dataset from [HuggingFace](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1) to your local `data/` folder.

You can download the dataset with the Hugging Face CLI tool. Please note that the full dataset requires 2.3TB of storage.

```shell
hf download embed2scale/SSL4EO-S12-v1.1 --repo-type dataset --local-dir data/SSL4EOS12
```

If you like to download only a subset of the data, you can specify it with `--include`.
```shell
# Only download val data
hf download embed2scale/SSL4EO-S12-v1.1 --repo-type dataset --include "val/*" --local-dir data/SSL4EOS12

# Only download a single modality (e.g., S2L2A)
hf download embed2scale/SSL4EO-S12-v1.1 --repo-type dataset --include "*/S2L2A/*" --local-dir data/SSL4EOS12
```

## Usage

We provide code for a PyTorch dataset in [ssl4eos12_dataset.py](ssl4eos12_dataset.py). You can initialize a data loader with the following code:

Standardization values:
```json
{
  "S2L1C": {
    "mean": [1607.345, 1393.068, 1320.225, 1373.963, 1562.536, 2110.071, 2392.832, 2321.154, 2583.77,  838.712, 21.753, 2205.112, 1545.798],
    "std": [786.523, 849.702, 875.318, 1143.578, 1126.248, 1161.98, 1273.505, 1246.79, 1342.755, 576.795, 45.626, 1340.347, 1145.036]
  },
  "S2L2A": {
    "mean": [793.243, 924.863, 1184.553, 1340.936, 1671.402, 2240.082, 2468.412, 2563.244, 2627.704, 2711.071, 2416.714, 1849.625],
    "std": [1160.144, 1201.092, 1219.943, 1397.225, 1400.035, 1373.136, 1429.17, 1485.025, 1447.836, 1652.703, 1471.002, 1365.307]
  },
  "S2RGB": {
    "mean": [100.708, 87.489, 61.932],
    "std": [68.550, 47.647, 40.592]
  },
  "S1GRD": {
    "mean": [-12.577, -20.265],
    "std": [5.179, 5.872]
  }
}
```

[//]: # (TODO Add stats for other modalities.)

### Zarr chunk file version

We released a previous version of SSL4EO-S12 v1.1 using Zarr chunk files with 64 samples each. The version ist still available at [embed2scale/SSL4EO-S12-v1.1-Zarr](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1-Zarr).
We moved on to a webdataset version for better usability. 

[zarr_dataset.py](zarr_dataset.py) provides data loading code for previous version and the chunk file version is directly compatible with [TerraTorch's](https://terrastackai.github.io/terratorch/stable/) `GenericMultiModalDataModule` which is showcased in the config [terratorch_zarr_ssl4eos12.yaml](terratorch_zarr_ssl4eos12.yaml). 

## License

This repository is released under the Apache 2.0 license. The dataset is released under the CC-BY-4.0 license.

## Citation

If you use this dataset in your work, please cite:
```txt
@article{blumenstiel2025ssl4eos12,
  title={{SSL4EO-S12} v1.1: A Multimodal, Multiseasonal Dataset for Pretraining, Updated},
  author={Blumenstiel, Benedikt and Ait Ali Braham, Nassim and Albrecht, Conrad M and Maurogiovanni, Stefano and Fraccaro, Paolo},
  journal={arXiv preprint arXiv:2503.00168},
  year={2025}
}
```

This dataset is an updated version of:
```text
@article{wang2022ssl4eo,
  title={{SSL4EO-S12}: A large-scale multimodal, multitemporal dataset for self-supervised learning in Earth observation [Software and Data Sets]},
  author={Wang, Yi and Ait Ali Braham, Nassim and Xiong, Zhitong and Liu, Chenying and Albrecht, Conrad M and Zhu, Xiao Xiang},
  journal={IEEE Geoscience and Remote Sensing Magazine},
  volume={11},
  number={3},
  pages={98--106},
  year={2023},
  publisher={IEEE}
}
```