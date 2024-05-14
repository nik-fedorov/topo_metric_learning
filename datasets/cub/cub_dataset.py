import numpy as np
import pandas as pd
from pathlib import Path
import subprocess

from torch.utils.data import Dataset

from oml.datasets.base import DatasetWithLabels


class CUB_200_2011(Dataset):
    def __init__(self, transforms, is_train):
        dataset_folder = Path(__file__).parent
        dataset_path = dataset_folder / 'CUB_200_2011'
        if not dataset_path.exists():
            print(f'Dataset CUB_200_2011 is not found on path {dataset_path.absolute()}. Start downloading it...')
            subprocess.run(['bash', str(dataset_folder / 'download_cub.sh')], cwd=str(dataset_folder))
            print('Dataset CUB_200_2011 is successfully downloaded and converted!')

        df = pd.read_csv(str(dataset_path / 'df.csv'))
        # use trainval split as in DML papers
        df[['is_query', 'is_gallery']] = np.nan
        df.loc[df['label'] <= 100, 'split'] = 'train'
        df.loc[df['label'] > 100, 'split'] = 'validation'
        df.loc[df['label'] > 100, ['is_query', 'is_gallery']] = True

        if is_train:
            self.dataset = DatasetWithLabels(df[df['split'] == 'train'], transform=transforms, dataset_root=str(dataset_path))
        else:
            self.dataset = DatasetWithLabels(df[df['split'] == 'validation'], transform=transforms, dataset_root=str(dataset_path))

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def get_labels(self):
        return self.dataset.get_labels()
