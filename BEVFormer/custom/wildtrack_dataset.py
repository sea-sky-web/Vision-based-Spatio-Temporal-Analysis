import os
import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.builder import DATASETS

@DATASETS.register_module()
class WildTrackDataset(Custom3DDataset):
    """WildTrack数据集加载器，与MMDetection3D框架兼容"""

    CLASSES = ('Pedestrian',)

    def __init__(self, data_root, ann_file, pipeline=None, test_mode=False, **kwargs):
        self.data_root = data_root
        self.annotation_path = os.path.join(self.data_root, 'annotations_positions/')
        self.calib_path = os.path.join(self.data_root, 'calibrations/')
        self.num_cams = 7
        # Wildtrack有400帧
        self.frame_indices = list(range(400))
        
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=self.CLASSES,
            test_mode=test_mode,
            **kwargs
        )

    def _parse_xml_matrix(self, xml_file, data_tag):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        data_element = root.find(f".//{data_tag}")
        rows = int(data_element.find("rows").text)
        cols = int(data_element.find("cols").text)
        data = list(map(float, data_element.find("data").text.strip().split()))
        return np.array(data).reshape(rows, cols)

    def _get_calib_info(self):
        """一次性加载并缓存所有相机的内外参"""
        intrinsics = []
        extrinsics = []
        for i in range(1, self.num_cams + 1):
            intr_path = os.path.join(self.calib_path, 'intrinsic_zero', f'intrinsic_C{i}.xml')
            extr_path = os.path.join(self.calib_path, 'extrinsic', f'extrinsic_C{i}.xml')
            
            # 加载内参
            K = self._parse_xml_matrix(intr_path, 'intrinsic_matrix')
            intrinsics.append(K)

            # 加载外参 (cam2world)
            R = self._parse_xml_matrix(extr_path, 'rotation_matrix')
            T = self._parse_xml_matrix(extr_path, 'translation_vector').flatten()

            # 求逆得到世界到相机的转换 (world2cam)
            R_inv = R.T
            T_inv = -R.T @ T

            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R_inv
            extrinsic[:3, 3] = T_inv
            extrinsics.append(extrinsic)

        return intrinsics, extrinsics

    def load_annotations(self, ann_file):
        """加载标注文件，这里我们为每一帧生成一个数据项"""
        # ann_file 在此场景下未使用，因为标注是按帧存储的
        data_infos = []
        intrinsics, extrinsics = self._get_calib_info()

        for frame_id in self.frame_indices:
            info = {
                'frame_id': frame_id,
                'img_filenames': [os.path.join(self.data_root, 'Image_subsets', f'C{i}', f'{frame_id:08d}.png') for i in range(1, self.num_cams + 1)],
                'cam_intrinsics': intrinsics,
                'cam_extrinsics': extrinsics
            }

            # 加载3D标注
            ann_path = os.path.join(self.annotation_path, f'{frame_id:08d}.json')
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    ann_data = json.load(f)

                gt_bboxes_3d = []
                gt_labels_3d = []
                for person in ann_data:
                    # BEVFormer需要(x, y, z, w, l, h, yaw)格式
                    # Wildtrack只提供(x, y, z)，我们假设一个固定的包围盒尺寸和方向
                    x, y, z = person['position3D']
                    w, l, h = 0.8, 0.8, 1.8  # 行人平均尺寸
                    yaw = 0  # 假设方向为0
                    gt_bboxes_3d.append([x, y, z, w, l, h, yaw])
                    gt_labels_3d.append(0) # 'Pedestrian' 类别

                info['ann_info'] = {
                    'gt_bboxes_3d': np.array(gt_bboxes_3d, dtype=np.float32),
                    'gt_labels_3d': np.array(gt_labels_3d, dtype=np.int64)
                }
            data_infos.append(info)

        return data_infos

    def get_data_info(self, index):
        """根据索引获取数据信息"""
        info = self.data_infos[index]

        # 准备BEVFormer需要的多视图格式
        input_dict = {
            'sample_idx': info['frame_id'],
            'img_filename': info['img_filenames'],
            'cam_intrinsic': info['cam_intrinsics'],
            'lidar2cam': info['cam_extrinsics'], # 在BEVFormer中，外参通常指 lidar/world to cam
            'ann_info': info.get('ann_info')
        }
        return input_dict