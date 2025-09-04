import os
import cv2
import numpy as np
import torch

# 假设BEVFormer的NuScenesDataset类已经存在
# 实际使用时需要确保正确导入
class CustomWildTrackDataset:
    """自定义WildTrack数据集类，用于加载WildTrack数据集的图像和相机参数"""
    
    def __init__(self, data_root='data/Wildtrack/', **kwargs):
        """初始化数据集
        
        Args:
            data_root: WildTrack数据集的根目录
        """
        self.data_root = data_root
        self.calib_path = os.path.join(self.data_root, 'calibrations/')
        
    def prepare_test_data(self, idx):
        """准备测试数据
        
        Args:
            idx: 帧索引
            
        Returns:
            dict: 包含图像和相机参数的字典
        """
        info = {
            'img_metas': {
                'cam_intrinsic': self.load_intrinsics(),
                'cam_extrinsic': self.load_extrinsics(),
                'num_cams': 7
            },
            'img': self.load_images(idx)
        }
        return info
    
    def load_images(self, frame_id=0):
        """加载WildTrack第frame_id帧的7张图像
        
        Args:
            frame_id: 帧索引，默认为0
            
        Returns:
            np.array: 形状为[7, H, W, 3]的图像数组
        """
        images = []
        for i in range(1, 8):  # WildTrack有7个摄像头，编号从1到7
            path = f"{self.data_root}/Image_subsets/C{i}/{frame_id:08d}.png"
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"无法加载图像: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        return np.stack(images)
    
    def load_intrinsics(self):
        """加载相机内参
        
        Returns:
            list: 包含7个相机内参矩阵的列表
        """
        intrinsics = []
        for i in range(1, 8):
            # 假设内参文件格式为stereo_calibrations_intrinsic.txt或类似格式
            # 实际使用时需要根据WildTrack数据集的实际格式进行调整
            intr_path = os.path.join(self.calib_path, 'intrinsic_zero', f'intrinsic_C{i}.xml')
            # 这里简化处理，实际应该解析XML文件
            # 使用默认内参作为示例
            K = np.array([
                [1000.0, 0.0, 320.0],
                [0.0, 1000.0, 180.0],
                [0.0, 0.0, 1.0]
            ])
            intrinsics.append(torch.tensor(K, dtype=torch.float32))
        return intrinsics
    
    def load_extrinsics(self):
        """加载相机外参
        
        Returns:
            dict: 包含相机外参的字典
        """
        extrinsics = {}
        for i in range(1, 8):
            # 假设外参文件格式为extrinsic.txt或类似格式
            # 实际使用时需要根据WildTrack数据集的实际格式进行调整
            extr_path = os.path.join(self.calib_path, 'extrinsic', f'extrinsic_C{i}.xml')
            # 这里简化处理，实际应该解析XML文件
            # 使用默认外参作为示例
            R = np.eye(3)  # 旋转矩阵，默认为单位矩阵
            T = np.array([0.0, 0.0, 0.0])  # 平移向量，默认为零向量
            
            extrinsics[f'C{i}'] = {
                'R': torch.tensor(R, dtype=torch.float32),
                'T': torch.tensor(T, dtype=torch.float32)
            }
        return extrinsics