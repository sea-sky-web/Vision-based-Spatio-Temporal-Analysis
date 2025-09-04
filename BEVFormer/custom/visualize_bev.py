import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_bev(bev_feature, save_path='test_bev.png'):
    """可视化BEV特征图
    
    Args:
        bev_feature: BEV特征图，形状为[B, C, H, W]
        save_path: 保存路径，默认为'test_bev.png'
    """
    # 简单可视化：对通道维度取平均值，生成热图
    if isinstance(bev_feature, torch.Tensor):
        bev_mean = bev_feature[0].mean(dim=0).detach().cpu().numpy()
    else:
        bev_mean = bev_feature[0].mean(axis=0)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(bev_mean, cmap='viridis')
    plt.colorbar(label='Feature Intensity')
    plt.title('BEV Feature Map')
    plt.savefig(save_path)
    plt.close()
    
    print(f"BEV特征图已保存至 {save_path}")
    
    return bev_mean

def visualize_bev_with_annotations(bev_feature, annotations=None, save_path='test_bev_with_annotations.png'):
    """可视化BEV特征图并叠加标注
    
    Args:
        bev_feature: BEV特征图，形状为[B, C, H, W]
        annotations: 标注信息，格式为[(x, y), ...]
        save_path: 保存路径，默认为'test_bev_with_annotations.png'
    """
    bev_mean = visualize_bev(bev_feature, save_path)
    
    if annotations is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(bev_mean, cmap='viridis')
        plt.colorbar(label='Feature Intensity')
        
        # 绘制标注点
        for x, y in annotations:
            plt.plot(x, y, 'ro', markersize=8)
            
        plt.title('BEV Feature Map with Annotations')
        plt.savefig(save_path)
        plt.close()
        
        print(f"带标注的BEV特征图已保存至 {save_path}")