# SSL4EO-S12 v1.1

SSL4EO-S12 v1.1 is an updated version of the popular EO pre-training dataset [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12).
Read more about the reasons behind our update and further improvements in our technical report on [arXiv](https://arxiv.org/abs/2503.00168).  

## Data

The dataset includes 246,144 locations with four timestamps each from the modalities S2L1C, S2L2A, S1GRD and S2RGB. 
We refer to our [technical report](https://arxiv.org/abs/2503.00168) for details. 

![ssl4eos12_samples.png](assets%2Fssl4eos12_samples.png)

The samples are stored in 3,846 Zarr Zip files (zarr version 2) that enable efficient storage and data loading. 
The metadata is stored directly with the samples as additional data variables.
Each Zarr files contains 64 samples (unique locations) with four timestamps each. 
The timestamps are chunked separately, which enables efficient loading of single timestamps.

You can read a Zarr file with:
```python
import xarray as xr
ds = xr.open_zarr('filename.zarr.zip')  # load xarray dataset
data = ds.bands.values  # load numpy array with dims [B, T, C, H, W]
```
Zarr was recently updated to version 3 which might lead to errors. You can fix easily by installing `zarr==2.18.0`.

Example of a S2L2A xarray dataset:
```text
<xarray.Dataset> Size: 446MB
Dimensions:     (band: 12, sample: 64, time: 4, y: 264, x: 264)
Coordinates:
  * band        (band) <U3 144B 'B01' 'B02' 'B03' 'B04' ... 'B09' 'B11' 'B12'
  * sample      (sample) <U7 2kB '0080717' '0060573' ... '0179869' '0012333'
  * time        (time) int64 32B 0 1 2 3
  * x           (x) int64 2kB 0 1 2 3 4 5 6 7 ... 257 258 259 260 261 262 263
  * y           (y) int64 2kB 0 1 2 3 4 5 6 7 ... 257 258 259 260 261 262 263
Data variables:
    bands       (sample, time, band, y, x) int16 428MB 1463 1457 ... 1777 1673
    center_lat  (sample) float64 512B 42.66 -30.64 50.47 ... 27.29 -23.06 29.99
    center_lon  (sample) float64 512B 125.6 121.4 128.3 ... -104.8 43.64 48.18
    cloud_mask  (sample, time, y, x) uint8 18MB 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0
    crs         (sample) int64 512B 32651 32751 32652 ... 32613 32738 32639
    file_id     (sample, time) <U38 39kB '20201116T023001_20201116T022955_T51...
    sample_id   (sample, time) <U9 9kB '0080717_0' '0080717_1' ... '0012333_3'
    time_       (sample, time) datetime64[ns] 2kB 2020-11-16T02:30:01 ... 202...
    x_          (sample, x) float64 135kB 7.149e+05 7.149e+05 ... 2.289e+05
    y_          (sample, y) float64 135kB 4.728e+06 4.728e+06 ... 3.319e+06
```

Example of a S1GRD xarray dataset:
```text
<xarray.Dataset> Size: 72MB
Dimensions:     (band: 2, sample: 64, time: 4, y: 264, x: 264)
Coordinates:
  * band        (band) <U2 16B 'vv' 'vh'
  * sample      (sample) <U7 2kB '0080717' '0060573' ... '0179869' '0012333'
  * time        (time) int64 32B 0 1 2 3
  * x           (x) int64 2kB 0 1 2 3 4 5 6 7 ... 257 258 259 260 261 262 263
  * y           (y) int64 2kB 0 1 2 3 4 5 6 7 ... 257 258 259 260 261 262 263
Data variables:
    bands       (sample, time, band, y, x) float16 71MB -15.13 -17.44 ... -29.0
    center_lat  (sample) float64 512B 42.66 -30.64 50.47 ... 27.29 -23.06 29.99
    center_lon  (sample) float64 512B 125.6 121.4 128.3 ... -104.8 43.64 48.18
    crs         (sample) int64 512B 32651 32751 32652 ... 32613 32738 32639
    file_id     (sample, time) <U67 69kB 'S1B_IW_GRDH_1SDV_20201127T214646_20...
    sample_id   (sample, time) <U9 9kB '0080717_0' '0080717_1' ... '0012333_3'
    time_       (sample, time) datetime64[ns] 2kB 2020-11-27T21:46:46 ... 202...
    x_          (sample, x) float64 135kB 7.149e+05 7.149e+05 ... 2.289e+05
    y_          (sample, y) float64 135kB 4.728e+06 4.728e+06 ... 3.319e+06
```

We provide parquet files with some metadata based on S2 in [val_metadata.parquet](splits%2Fval_metadata.parquet) and [train_metadata.parquet](splits%2Ftrain_metadata.parquet) which you can read with geopandas.

Example from the validation set: 
```text
sample_id                                              0000173
time_id                                                      0
time                                       2020-04-19 09:40:31
tile_id                                                 T33SWC
geometry     POLYGON ((15.655119659160581 38.06148945118448...
crs                                                      32633
file               ssl4eos12_val_seasonal_data_000022.zarr.zip
Name: 0000173_0
```

## Download

You can download the data from [Julich DataHub](https://datapub.fz-juelich.de/ssl4eo-s12/) with the following script:

```shell
# Download all data
wget --recursive --no-parent --reject "index.html*" --execute robots=off -nH -P data https://datapub.fz-juelich.de/ssl4eo-s12/
```

The script will download the data to your `data/` folder with the following format:
```text
data/
└── ssl4eo-s12
    ├── train
    │   ├── S1GRD
    │   │   ├── ssl4eos12_train_seasonal_data_000001.zarr.zip
    │   │   ├── ssl4eos12_train_seasonal_data_000002.zarr.zip
    │   │   ├── ...
    │   │   └── ssl4eos12_train_seasonal_data_003812.zarr.zip
    │   ├── S2L1C
    │   ├── S2L2A
    │   └── S2RGB
    └── val
        ├── S1GRD
        │   ├── ssl4eos12_val_seasonal_data_000001.zarr.zip
        │   ├── ssl4eos12_val_seasonal_data_000002.zarr.zip
        │   ├── ...
        │   └── ssl4eos12_val_seasonal_data_000034.zarr.zip
        ├── S2L1C
        ├── S2L2A
        └── S2RGB
```

You can specify a subdirectory for downloading a subset of the data:
```shell
# Download validation data
wget --recursive --no-parent --reject "index.html*" --execute robots=off -nH -P data https://datapub.fz-juelich.de/ssl4eo-s12/val/
# Download S2L2A validation data
wget --recursive --no-parent --reject "index.html*" --execute robots=off -nH -P data https://datapub.fz-juelich.de/ssl4eo-s12/val/S2L2A/
```

## Usage

We provide code for a PyTorch dataset in [ssl4eos12_dataset.py](ssl4eos12_dataset.py). You can initialize a data loader with the following code:
```python
from torch.utils.data import DataLoader
from torchvision import transforms
from ssl4eos12_dataset import SSL4EOS12Dataset, collate_fn, S2L1C_MEAN, S2L1C_STD, S2L2A_MEAN, S2L2A_STD, S1GRD_MEAN, S1GRD_STD

# We concatenate the modalities for the transform function. Depending on the parameter concat=True/False, 
# the data is returned as a concatenated tensor or split into the single modalities after the transform.
train_transform = transforms.Compose([
    transforms.RandomCrop(224),  # The data has size 264x264. We recommend RandomCrop for train and CenterCrop for val.
    transforms.Normalize(mean=S2L1C_MEAN + S2L2A_MEAN + S1GRD_MEAN, std=S2L1C_STD + S2L2A_STD + S1GRD_STD)
    # Data is loaded as torch Tensor, so no ToTensor() needed.
])

train_dataset = SSL4EOS12Dataset(
    data_dir='data/ssl4eo-s12/train',
    split_file='data/ssl4eo-s12/splits/ssl4eos12_train.txt',  # optional, speeds up the initialization.
    modalities=['S2L1C', 'S2L2A', 'S1GRD'], # optional, list of modality folders.
    transform=train_transform,  # optional, torchvision transforms. Returns tensors if not provided.
    concat=False,  # Concatenate all modalities along the band dimension.
    single_timestamp=False,  # Load single timestamps rather than time series.
    num_batch_samples=64,  # optional, subsample samples in each zarr file.
)

train_loader  = DataLoader(
    dataset=train_dataset,
    batch_size=1,  # Note that each batch file contains already 64 samples!
    shuffle=True,
    collate_fn=collate_fn,  # Data needs to be concatenated along sample dimension instead of being stacked
)
```

Alternatively, you can use the `GenericMultiModalDataModule` from [TerraTorch](https://github.com/IBM/terratorch) if you like to use TorchGeo or TerraTorch for your pre-training.
We provide an example config here: [terratorch_ssl4eos12.yaml](terratorch_ssl4eos12.yaml).

Standardization values:
```json
{
  "S2L1C": {
    "mean": [2607.345, 2393.068, 2320.225, 2373.963, 2562.536, 3110.071, 3392.832, 3321.154, 3583.77, 1838.712, 1021.753, 3205.112, 2545.798],
    "std": [786.523, 849.702, 875.318, 1143.578, 1126.248, 1161.98, 1273.505, 1246.79, 1342.755, 576.795, 45.626, 1340.347, 1145.036]
  },
  "S2L2A": {
    "mean": [1793.243, 1924.863, 2184.553, 2340.936, 2671.402, 3240.082, 3468.412, 3563.244, 3627.704, 3711.071, 3416.714, 2849.625],
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

## License

This repository is released under the Apache 2.0 license. The dataset is released under the CC-BY-4.0 license.

## Citation

If you use this dataset in your work, please cite:
```txt
@article{blumenstiel2025ssl4eos12,
  title={{SSL4EOS12 v1.1 – A Multimodal, Multiseasonal Dataset for Pretraining}},
  author={Blumenstiel, Benedikt and Braham, Nassim Ait Ali and Albrecht, Conrad M and Maurogiovanni, Stefano and Fraccaro, Paolo},
  journal={arXiv preprint arXiv:2503.00168},
  year={2025}
}
```

This dataset is an updated version of:
```text
@article{wang2022ssl4eo,
  title={{SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation}},
  author={Wang, Yi and Braham, Nassim Ait Ali and Xiong, Zhitong and Liu, Chenying and Albrecht, Conrad M and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:2211.07044},
  year={2022}
}
```