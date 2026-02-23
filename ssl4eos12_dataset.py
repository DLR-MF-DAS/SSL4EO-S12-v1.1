# Copyright 2025 DLR and IBM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file includes code adapted from work by EPFL and Apple Inc.,
# licensed under the Apache License, Version 2.0.
# Source: https://github.com/apple/ml-4m/
# and from IBM, licensed under the Apache License, Version 2.0.
# Source: https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1

import os
import io
import re
import zarr
import torch
import warnings
import fsspec
import braceexpand
import albumentations
import numpy as np
import webdataset as wds
from einops import rearrange
from collections.abc import Callable, Iterable
from torch.utils.data._utils.collate import default_collate
from webdataset.handlers import warn_and_continue

# Definition of all shard files in SSl4EOS12
split_files = {
        "train": "ssl4eos12_shard_{000001..000477}.tar",
        "val": "ssl4eos12_shard_{000001..000005}.tar",
    }

statistics = {
    "mean": {
        "S2L1C": [1607.345, 1393.068, 1320.225, 1373.963, 1562.536, 2110.071, 2392.832, 2321.154, 2583.77,  838.712, 21.753, 2205.112, 1545.798],
        "S2L2A": [793.243, 924.863, 1184.553, 1340.936, 1671.402, 2240.082, 2468.412, 2563.244, 2627.704, 2711.071, 2416.714, 1849.625],
        "S2RGB": [100.708, 87.489, 61.932],
        "S1GRD": [-12.577, -20.265],
        "NDVI": [0.327],
        "DEM": [435.726],
    },
    "std": {
        "S2L1C": [786.523, 849.702, 875.318, 1143.578, 1126.248, 1161.98, 1273.505, 1246.79, 1342.755, 576.795, 45.626, 1340.347, 1145.036],
        "S2L2A": [1160.144, 1201.092, 1219.943, 1397.225, 1400.035, 1373.136, 1429.17, 1485.025, 1447.836, 1652.703, 1471.002, 1365.307],
        "S2RGB": [68.550, 47.647, 40.592],
        "S1GRD": [5.179, 5.872],
        "NDVI": [0.322],
        "DEM": [560.326]
    }
}


def build_ssl4eos12_dataset(
        path: str = "https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1/resolve/main/",
        modalities: list[str] | str = None,
        split: str = "val",
        files: str | None = None,
        urls: str | None = None,
        transform: Callable = None,
        batch_size: int = 8,
        return_metadata: bool = False,
        reindex_seasonal: bool = False,
        shuffle: bool = None,
        shardshuffle: int = 100,
        deterministic: bool = False,
        seed: int = None,
        partial: bool = None,
        **kwargs,
):
    """
    Builds a dataset for SSL4EO-S12 v1.1, see https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1.

    :param path: URL or local path to dataset root that with data structure ./{split}/{modality}/shard_{id}.tar
    :param modalities: List of modalities or a single modality name
    :param split: Split name ("train", "val"). Default to "val".
    :param files: Files to use, defaults to ssl4eos12_shard_{000001..000477}.tar for train and
        ssl4eos12_shard_{000001..000005}.tar for val.
    :param urls: Specify custom shard urls instead of providing the path, modalities, and split.
    :param batch_size: Specify batch size to load batches instead of samples via webdataset (Recommended).
        It requires batch_size=None in the data loader constructor.
    :param transform: Transform function to apply to the data, use MultimodalTransforms.
    :param return_metadata: Load center coordinates, timestamp (ns as int) and cloud mask (if available).
    :param reindex_seasonal: Samples is sorted by date. If true, reindex samples to match quartal 1, Q2, Q3, and Q4.
    :param shuffle: Shuffle samples and shards. Default to True for train and False for val.
    :param shardshuffle: The number of shards to shuffle, or None. Defaults to 100.
    :param deterministic: Whether to use deterministic shuffling. Defaults to False.
    :param seed: Random seed for shuffling. Defaults to None which uses random seeds.
    :param kwargs: Optional keyword arguments for single-modality which are passed to WebDataset constructor.
    :param empty_check: Check if shards are empty. Defaults to False.
    :param partial: Load partial batch at the end. Defaults to False for train and True for val.
    :return: WebDataset (single modality) or DataPipeline (multiple modalities)
    """
    if len(modalities) == 1:
        # Single modality
        modalities = modalities[0]

    # No shuffle and partial load for val
    shuffle = shuffle if shuffle is not None else split != "val"
    partial = partial if partial is not None else split == "val"
    return_metadata = return_metadata or reindex_seasonal  # metadata required for seasonal reindexing
    shardshuffle = shardshuffle * shuffle

    if isinstance(modalities, str):
        # Build standard WebDataset for single modality
        dataset = build_wds_dataset(
            path=path,
            modality=modalities,
            split=split,
            files=files,
            urls=urls,
            batch_size=batch_size,
            transform=transform,
            return_metadata=return_metadata,
            reindex_seasonal=reindex_seasonal,
            shardshuffle=shardshuffle,
            deterministic=deterministic,
            seed=seed,
            partial=partial,
            **kwargs
        )
        return dataset

    else:
        if len(kwargs):
            warnings.warn(f"keyword arguments ({kwargs}) are ignored for multiple modalities.")

        # Build custom multi-modal dataset
        dataset = build_multimodal_dataset(
            path=path,
            modalities=modalities,
            split=split,
            files=files,
            urls=urls,
            batch_size=batch_size,
            transform=transform,
            return_metadata=return_metadata,
            reindex_seasonal=reindex_seasonal,
            shardshuffle=shardshuffle,
            deterministic=deterministic,
            seed=seed,
            partial=partial,
        )
        return dataset


def zarr_decoding(key, value):
    if key == "zarr.zip" or key.endswith(".zarr.zip"):
        mapper = fsspec.filesystem("zip", fo=io.BytesIO(value), block_size=None).get_mapper("")
        return zarr.open_consolidated(mapper, mode="r")["bands"][...]


def zarr_metadata_decoding(sample):
    for key, value in list(sample.items()):
        if key == "zarr.zip" or key.endswith(".zarr.zip"):
            mapper = fsspec.filesystem("zip", fo=io.BytesIO(value), block_size=None).get_mapper("")
            data = zarr.open_consolidated(mapper, mode="r")
            sample[key] = data["bands"][...]

            # Add metadata
            if "center_lon" not in sample.keys():  # Same center point for all modalities
                sample["center_lon"] = data["center_lon"][...]
                sample["center_lat"] = data["center_lat"][...]
            if "cloud_mask" in data and "cloud_mask" not in sample.keys():  # Same S2 mask in all optical modalities
                sample["cloud_mask"] = data["cloud_mask"][...][:, np.newaxis, ...]  # Add channel dim to mask
            if (data["time"][...] > 1e6).any():  # DEM has no valid timestamp (value = 0)
                time_key = "time" if key == "zarr.zip" else "time_" + key
                sample[time_key] = data["time"][...]  # Integer values of type "datetime64[ns]"
        elif isinstance(value, str):
            # Skip str data
            pass
        else:
            # Fallback to webdataset autodecoder
            sample[key] = next(wds.decode()([{key: value}]))[key]

    return sample


def reindex_by_season(sample):
    # Get index of months (ignoring years)
    time_keys = [k for k in sample.keys() if k.startswith("time")] # Assuming all modalities are temporally aligned
    timestamps = sample[time_keys[0]].astype("datetime64[ns]")
    months = (timestamps.astype("datetime64[M]") - timestamps.astype("datetime64[Y]")).astype(int) + 1
    idx = months.argsort()

    # Reindex all modalities and metadata
    for k, v in sample.items():
        if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == len(idx):
            sample[k] = v[idx]

    return sample

def identity(sample):
    """Identity function that does nothing."""
    return sample


def build_wds_dataset(
        path: str = "https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1/resolve/main/",
        modality: str = "S2L2A",
        split: str = "val",
        files: str | None = None,
        urls: str | None = None,
        batch_size: int = 8,
        transform: Callable = None,
        return_metadata: bool = False,
        reindex_seasonal: bool = False,
        shardshuffle: int = 100,
        deterministic: bool = False,
        seed: int = None,
        empty_check: bool = False,
        partial: bool = False,
        *args, **kwargs
):
    if urls is None:
        # Select split files
        urls = os.path.join(path, split, modality, files or split_files[split])

    if split == "val" and empty_check:
        # Setting empty_check to True to avoid errors because of few val shard files
        empty_check = False

    # Build dataset
    dataset = wds.WebDataset(
        urls,
        *args,
        shardshuffle=shardshuffle,
        detshuffle=deterministic,
        seed=seed,
        handler=warn_and_continue,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
        empty_check=empty_check,
        **kwargs
    )

    # Decode from bytes to numpy arrays, etc.
    dataset = dataset.map(zarr_metadata_decoding) if return_metadata else dataset.decode(zarr_decoding)

    # Rename modality to "image" and remove temporal dimension
    dataset = dataset.rename(image="zarr.zip")

    if reindex_seasonal:
        dataset = dataset.map(reindex_by_season)

    if transform is not None:
        dataset = dataset.map(transform)

    # Create batches
    if batch_size is not None:
        dataset = dataset.batched(batch_size, partial=partial)

    return dataset


def build_multimodal_dataset(
        path: str = "https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1/resolve/main/",
        modalities: list = None,
        split: str = "val",
        urls: str | None = None,
        files: str | None = None,
        batch_size: int = 8,
        transform: Callable = None,
        return_metadata: bool = False,
        reindex_seasonal: bool = False,
        shardshuffle: int = 100,
        deterministic: bool = False,
        seed: int = None,
        empty_check: bool = False,
        partial: bool = False,
):
    if modalities is None:
        modalities = ["S2L2A", "S2L1C", "S2RGB", "S1GRD", "DEM", "NDVI", "LULC"]  # Default
    if urls is None:
        urls = os.path.join(path, split, f"[{','.join(modalities)}]", files or split_files[split])

    dataset = wds.DataPipeline(
        wds.ResampledShards(urls, deterministic=deterministic, seed=seed, empty_check=empty_check)
        if shardshuffle else wds.SimpleShardList(urls),
        wds.split_by_node,
        wds.split_by_worker,
        # Extract individual samples from multi-modal tar files
        multi_tarfile_samples,
        wds.shuffle(shardshuffle, seed=seed),
        # Decode from bytes to numpy arrays, etc.
        (wds.map(zarr_metadata_decoding) if return_metadata else wds.decode(zarr_decoding)),
        wds.map(remove_extensions),
        (wds.map(reindex_by_season) if reindex_seasonal else wds.map(identity)),
        wds.map(transform) if transform is not None else wds.map(identity),
        wds.batched(batch_size, collation_fn=collate_fn, partial=partial),
    )

    return dataset


def collate_fn(batch):
    # Wrapper for debugging
    try:
        return default_collate(batch)
    except Exception as e:
        for s in batch:
            print(s["__key__"])
            print(s["__url__"])
            print(s.keys())
        raise e


def extract_modality_names(s):
    """
    Function from https://github.com/apple/ml-4m/blob/main/fourm/data/unified_datasets.py.
    """
    # Regular expression pattern to match anything enclosed in "{" and "}", and comma separated
    pattern = r"\{([^}]*)\}"
    match = re.search(pattern, s)
    return match.group(1).split(",") if match else []


def remove_extensions(sample):
    """
    Function from https://github.com/apple/ml-4m/blob/main/fourm/data/unified_datasets.py.

    In webdatasets, we identify the type of a given modality by adding an extension
    in the form f"{modality_name}.{modality_extension}", e.g. "rgb.jpg" or "caption.json".
    This function removes them and returns a dictionary of {f"{modality_name}": modality}.
    """
    return {os.path.splitext(k.replace(".zip", ""))[0]: v for k, v in sample.items()}


def multi_tarfile_samples(
        src_iter: Iterable[dict],
):
    """
    This function is adapted from https://github.com/apple/ml-4m/blob/main/fourm/data/unified_datasets.py.

    Webdataset does not support splitting up shards by modality, so we need to do this manually.
    Usually, we would need to save all modalities in the same tar file, e.g. shard_root_train/{00000..12345}.tar,
    where each shard contains 1000 samples and each sample contains all modalities.
    This is not flexible when adding new modalities, so we instead save each modality in a separate tar file,
    e.g. shard_root_train_rgb/{00000..12345}.tar, shard_root_train_caption/{00000..12345}.tar, etc., where each shard contains
    again 1000 samples, but each sample contains only one modality. All samples in all shards have to be aligned.

    This function takes an iterator over shard URLs, where we use brace expansion to specify multiple tar files per modality.
    E.g. shard_root_train_[rgb,caption]/00123.tar will be expanded to shard_root_train_rgb/00123.tar and shard_root_train_caption/00123.tar,
    and the samples from these two tar files will be combined into a single sample.

    Args:
        src_iter: Iterator over shards that *already brace expanded the shard numbers*,
            e.g. {"url": "shard_root_train_[rgb,caption]/00000.tar"}, {"url": "shard_root_train_[rgb,caption]/00001.tar"}, ...
            This function will also work when no square braces for multiple modalities are used, e.g. {"url": "shard_root_train/00000.tar"}, ...
            It can be a drop-in replacement for wds.tarfile_samples.

    Yields:
        Dictionary of aligned samples from all modalities.
    """

    for src in src_iter:

        # Multi tar file URLs use brace expansion with square braces
        multi_tar_urls = src["url"].translate(str.maketrans("[]", "{}"))
        modality_names = extract_modality_names(multi_tar_urls)
        multi_tar_urls = list(braceexpand.braceexpand(multi_tar_urls))

        # Create tar iterators for shards of all modalities
        tar_iters = [
            wds.tarfile_samples([{"url": tar_url}]) for tar_url in multi_tar_urls
        ]

        try:
            # Loop over these iterators in parallel and combine the tar files from different modalities
            for multi_tar_files in zip(*tar_iters):

                merged_dict = {}
                merged_dict["__key__"] = multi_tar_files[0]["__key__"]
                merged_dict["__url__"] = src["url"]

                for modality_name, modality_dict in zip(
                        modality_names, multi_tar_files
                ):
                    _key = modality_dict.pop("__key__")
                    _url = modality_dict.pop("__url__")

                    if _key != merged_dict["__key__"]:
                        raise ValueError(
                            f"Divergence detected! Trying to merge keys {_key} of {modality_name} and {merged_dict['__key__']} of merged_dict with modalities {merged_dict.keys()}."
                        )

                    for k, v in modality_dict.items():
                        if modality_name is None:
                            merged_dict[k] = v
                        else:
                            merged_dict[f"{modality_name}.{k}"] = v

                yield merged_dict

        except Exception as e:
            warnings.warn(f"Exception occurred while processing {src['url']}: {repr(e)}."
                          f"Skipping shard")
            continue


class Transpose(albumentations.ImageOnlyTransform):
    """
    Rearrange is a generic image transformation that reshapes an input tensor using a custom einops pattern.

    This transform allows flexible reordering of tensor dimensions based on the provided pattern and arguments.
    """

    def __init__(self, axis: list):
        """
        Initialize the Transpose transform.

        Args:
            axis (list): Axis for numpy.transpose.
        """
        super().__init__(p=1)
        self.axis = axis

    def apply(self, img, **params):
        return np.transpose(img, self.axis)

    def get_transform_init_args_names(self):
        return "transpose"


def default_non_image_transform(array):
    if hasattr(array, "dtype") and (array.dtype == float or array.dtype == int):
        return torch.from_numpy(array.copy())
    else:
        return array


class MultimodalTransforms:
    """
    MultimodalTransforms applies albumentations transforms to multiple image modalities.

    This class supports both shared transformations across modalities and separate transformations for each modality.
    It also handles non-image modalities by applying a specified non-image transform.

    This code is adapted from https://github.com/IBM/terratorch/blob/main/terratorch/datasets/transforms.py.
    """

    def __init__(
            self,
            transforms: dict | albumentations.Compose,
            non_image_modalities: list[str] | None = None,
            non_image_transforms: object | None = None,
    ):
        """
        Initialize the MultimodalTransforms.

        Args:
            transforms (dict or A.Compose): The transformation(s) to apply to the data.
            non_image_modalities (list[str] | None): List of keys corresponding to non-image modalities.
            non_image_transforms (object | None): A transform to apply to non-image modalities.
                If None, a default transform is used.
        """
        self.transforms = transforms
        self.non_image_modalities = non_image_modalities or []
        self.non_image_transforms = non_image_transforms or default_non_image_transform

    def __call__(self, data: dict):
        # albumentations requires a key "image" and treats all other keys as additional targets
        image_modality = "image" if "image" in data else \
            [k for k in data.keys() if k not in self.non_image_modalities][0]  # Find an image modality name
        data["image"] = data[image_modality]  # albumentations expects an input called "image"
        data = self.transforms(**data)
        if image_modality != "image":
            _ = data.pop("image")

        # Process sequence data which is ignored by albumentations as "global_label"
        for modality in self.non_image_modalities:
            if modality in data:
                data[modality] = self.non_image_transforms(data[modality])

        return data


class FlattenTemporalIntoChannels(albumentations.ImageOnlyTransform):
    """
    Code adapted from https://github.com/terrastackai/terratorch/blob/main/terratorch/datasets/transforms.py

    FlattenTemporalIntoChannels is an image transformation that flattens the temporal dimension into the channel dimension.

    This transform rearranges an input tensor with a temporal dimension into one where the time and channel dimensions
    are merged. It expects the input to have a fixed number of dimensions defined by N_DIMS_FOR_TEMPORAL.
    """
    def __init__(self):
        """
        Initialize the FlattenTemporalIntoChannels transform.
        """
        super().__init__(True, 1)

    def apply(self, img, **params):
        rearranged = rearrange(img, "time height width channels -> height width (time channels)")
        return rearranged

    def get_transform_init_args_names(self):
        return ()


class UnflattenTemporalFromChannels(albumentations.ImageOnlyTransform):
    """
    Code adapted from https://github.com/terrastackai/terratorch/blob/main/terratorch/datasets/transforms.py

    UnflattenTemporalFromChannels is an image transformation that restores the temporal dimension from the channel dimension.

    This transform is typically applied after converting images to a channels-first format (e.g., after ToTensorV2)
    and rearranges the flattened temporal information back into separate time and channel dimensions.
    """

    def __init__(self, n_timesteps: int):
        super().__init__(True, 1)
        self.n_timesteps = n_timesteps

    def apply(self, img, **params):
        if img.shape[-1] < self.n_timesteps:
            time = 1 # Assuming missing timesteps (e.g., DEM)
        else:
            time = self.n_timesteps

        rearranged = rearrange(
            img, "(time channels) height width -> time channels height width", time=time
        )
        return rearranged

    def get_transform_init_args_names(self):
        return "n_timesteps"


class MultimodalNormalize(albumentations.ImageOnlyTransform):
    def __init__(self, mean: dict[str, list[float]], std: dict[str, list[float]]):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, **batch):
        for m in self.mean.keys():
            if m not in batch.keys():
                continue
            batch[m] = (batch[m] - self.mean[m]) / self.std[m]
        return batch
