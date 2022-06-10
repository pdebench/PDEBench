import h5py
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from pytorch_lightning import LightningDataModule


class HDF5Dataset(Dataset):
    """hdf5 dataset, generated from phiflow model
    :param dir_path: the directory path of saved .h5 files
    :param transform: the transforms we want to apply on data
    """

    def __init__(self, dir_path, transform=None):
        super().__init__()
        path = Path(dir_path)
        assert path.is_dir()
        files_path = list(path.glob('*.h5'))  # all .h5 files' path
        assert len(files_path) > 0

        self.data_info = {}
        self.transform = transform
        self.count = []
        self.config = []
        self.names = []

        for files_path in files_path:
            with h5py.File(str(files_path.resolve())) as f:
                config = f.attrs.get('config')
                for ds_name, ds in f.items():
                    self.names.append(ds_name)
                    b = ds.shape[0]
                    if ds_name not in self.data_info:
                        self.data_info[ds_name] = [ds[...]]
                    else:
                        self.data_info[ds_name].append(ds[...])
                last_count = self.count[-1] if len(self.count) > 0 else 0
                self.count.append(last_count + b)
                self.config.append(config)

    def __len__(self):
        return self.count[-1]

    def __getitem__(self, index):
        data, config = self._load_data(index)
        if self.transform:
            data = [self.transform(d) for d in data]
        return data, config

    def _load_data(self, idx):
        ds = []
        for n in self.count:
            if n >= idx + 1:
                batch_idx = self.count.index(n)
                break
        last_count = self.count[batch_idx - 1] if batch_idx > 0 else 0
        for ds_list in self.data_info.values():
            ds.append(ds_list[batch_idx][idx - last_count])
        return ds, self.config[batch_idx]


# PATH_DATASETS = 'dummy_dataset'

class HDF5DatasetLightning(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, transforms=None):
        super().__init__()
        self.train = None
        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = HDF5Dataset(self.data_dir, transform=self.transforms)

    def train_dataloader(self):
        print(self.train is None)
        return DataLoader(self.train, batch_size=self.batch_size)


if __name__ == "__main__":
    dir_path = 'download_dataset'  # random_force_field--ns_sim--10.h5 in this directory

    # test pytorch dataset
    dataset = HDF5Dataset(dir_path=dir_path, transform=None)
    names = dataset.names
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    data, config = next(iter(dataloader))
    for i, d in enumerate(data):
        print(f'{names[i].upper()} batched data shape: ', d.size())
    print('number of config files: ', len(config))

    # test pytorch lightning dataset
    lightning_dataset = HDF5DatasetLightning(dir_path, batch_size=64, transforms=None)
    lightning_dataset.setup()
    lightning_dataloader = lightning_dataset.train_dataloader()
    data, config = next(iter(lightning_dataloader))
    for i, d in enumerate(data):
        print(f'{names[i].upper()} batched data shape: ', d.size())
    print('number of config files: ', len(config))
