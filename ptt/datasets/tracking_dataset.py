import numpy as np
import torch.utils.data as torch_data

from pathlib import Path
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor_tracking import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger.info if logger is not None else print
        if self.dataset_cfg is None or class_names is None: return

        self.split = dataset_cfg.DATA_SPLIT[self.mode]
        self.load_from_db = dataset_cfg.LOAD_FROM_DATABASE
        self.debug = dataset_cfg.get('DEBUG', False)
        self.ref_coor = dataset_cfg.REF_COOR.upper()
        self.root_path = root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)
        self.point_cloud_range = np.array(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)

        if self.dataset_cfg.get('POINT_FEATURE_ENCODING') is not None:
            self.point_feature_encoder = PointFeatureEncoder(
                self.dataset_cfg.POINT_FEATURE_ENCODING
            )
        else:
            self.point_feature_encoder = None

        if self.dataset_cfg.get('DATA_AUGMENTOR') is not None:
            self.data_augmentor = DataAugmentor(
                self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
            ) if self.training else None
        else:
            self.data_augmentor = None

        if self.dataset_cfg.get('DATA_PROCESSOR') is not None:
            self.data_processor = DataProcessor(
                self.dataset_cfg.DATA_PROCESSOR,
                training=self.training,
            )
        else:
            self.data_processor = None

        self.grid_size = self.data_processor.grid_size if self.data_processor else -1
        self.voxel_size = self.data_processor.voxel_size if self.data_processor else -1
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
