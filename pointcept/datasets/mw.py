"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import os
from copy import deepcopy

import numpy as np
import open3d as o3d
from pointcept.utils.logger import get_root_logger
from .defaults import DefaultDataset

from .builder import DATASETS
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class MWDataset(DefaultDataset):
    def __init__(
            self,
            split="test",
            data_root="data/mw",
            transform=None,
            test_mode=False,
            test_cfg=None,
            cache=False,
            loop=1,
    ):
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )


    def get_data_list(self):
        data_list = glob.glob(os.path.join(self.data_root, "*.05.ply"))

        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]

        pcd = o3d.io.read_point_cloud(data_path)
        data = {'coord': np.asarray(pcd.points),
                'color': np.asarray(pcd.colors)
                }

        name = os.path.basename(self.data_list[idx % len(self.data_list)])

        scene_id = data_path
        coord = data["coord"]
        color = data["color"]
        segment = np.ones(coord.shape[0]) * -1
        instance = np.ones(coord.shape[0]) * -1

        data_dict = dict(
            name=name,
            coord=coord,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )

        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(segment=data_dict.pop("segment"), name=self.get_data_name(idx))

        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list

        return result_dict

    def __getitem__(self, idx):
        return self.prepare_test_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
