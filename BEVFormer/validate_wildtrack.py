import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# 添加自定义模块路径
sys.path.append(os.path.abspath('.'))

# 导入自定义模块
from custom.wildtrack_dataset import CustomWildTrackDataset
from custom.visualize_bev import visualize_bev, visualize_bev_with_annotations

def main():
    """验证BEVFormer在WildTrack数据集上的性能"""
    print("开始验证BEVFormer在WildTrack数据集上的性能...")
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 配置文件和检查点路径
    config_file = 'projects/configs/bevformer/custom_wildtrack.py'
    checkpoint_file = 'bevformer_base_epoch_24.pth'
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        print(f"错误: 配置文件 {config_file} 不存在")
        return
    
    if not os.path.exists(checkpoint_file):
        print(f"错误: 检查点文件 {checkpoint_file} 不存在")
        print("请下载预训练模型: https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_base_epoch_24.pth")
        return
    
    try:
        # 导入mmdet3d相关模块
        from mmdet3d.apis import init_model
    except ImportError:
        print("错误: 无法导入mmdet3d模块，请确保已正确安装")
        return
    
    try:
        # 初始化模型
        print("正在加载模型...")
        model = init_model(config_file, checkpoint_file, device=device)
        print("模型加载成功")
        
        # 创建数据集实例
        print("正在加载WildTrack数据集...")
        dataset = CustomWildTrackDataset(data_root='data/Wildtrack/')
        
        # 准备测试数据（第一帧）
        print("正在准备测试数据...")
        frame_id = 0  # 使用第一帧进行测试
        data = dataset.prepare_test_data(frame_id)
        
        # 将数据移动到设备上
        if isinstance(data['img'], np.ndarray):
            data['img'] = torch.from_numpy(data['img']).permute(0, 3, 1, 2).float().to(device)
        
        # 提取BEV特征
        print("正在提取BEV特征...")
        with torch.no_grad():
            # 前向传播，获取BEV特征
            # 注意：这里的实现可能需要根据BEVFormer的实际API进行调整
            result = model.extract_feat(img=data['img'], img_metas=data['img_metas'])
            
            # 假设bev_feature是一个字典，包含'bev_feature'键
            if isinstance(bev_feature, dict) and 'bev_feature' in bev_feature:
                bev_feature = bev_feature['bev_feature']
            elif isinstance(bev_feature, tuple) and len(bev_feature) > 0:
                # 如果是元组，取最后一个元素作为BEV特征
                bev_feature = bev_feature[-1]
        
        # 可视化BEV特征
        print("正在可视化BEV特征...")
        visualize_bev(bev_feature, save_path='test_bev.png')
        
        print("验证完成！请查看test_bev.png文件以检查BEV特征图")
        
    except Exception as e:
        print(f"验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()