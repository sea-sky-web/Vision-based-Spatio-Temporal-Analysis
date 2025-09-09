import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_bev_with_annotations(bev_feature, annotations, pc_range, bev_height, bev_width, save_path='test_bev_with_annotations.png'):
    """
    可视化BEV特征图并叠加3D真实标注。

    Args:
        bev_feature (torch.Tensor): BEV特征图，形状为 [1, C, H, W]。
        annotations (list): 3D标注列表，每个元素是 [x, y, z] 坐标。
        pc_range (list): BEV空间范围 [min_x, min_y, min_z, max_x, max_y, max_z]。
        bev_height (int): BEV特征图的高度。
        bev_width (int): BEV特征图的宽度。
        save_path (str): 图像保存路径。
    """
    # 1. 生成BEV热力图
    # 对通道维度取平均值以生成2D热力图
    if isinstance(bev_feature, torch.Tensor):
        bev_heatmap = bev_feature.squeeze(0).mean(dim=0).detach().cpu().numpy()
    else:
        bev_heatmap = np.mean(bev_feature.squeeze(0), axis=0)

    # 2. 转换3D标注到BEV像素坐标
    bev_coords = []
    for pos3d in annotations:
        x, y, z = pos3d
        # 检查点是否在BEV范围内
        if pc_range[0] <= x < pc_range[3] and pc_range[1] <= y < pc_range[4]:
            # 计算像素坐标
            coord_x = (x - pc_range[0]) / (pc_range[3] - pc_range[0]) * bev_width
            coord_y = (y - pc_range[1]) / (pc_range[4] - pc_range[1]) * bev_height

            # BEV y轴通常与世界坐标y轴方向相反
            bev_coords.append((coord_x, bev_height - coord_y))

    # 3. 绘制图像
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(bev_heatmap, cmap='viridis', origin='lower', extent=[pc_range[0], pc_range[3], pc_range[1], pc_range[4]])
    
    # 绘制标注点
    if bev_coords:
        coords_x, coords_y_transformed = zip(*bev_coords)
        
        # 将转换后的像素坐标映射回物理坐标以在图上正确显示
        plot_x = [pc_range[0] + (c / bev_width) * (pc_range[3] - pc_range[0]) for c in coords_x]
        plot_y = [pc_range[4] - (c / bev_height) * (pc_range[4] - pc_range[1]) for c in coords_y_transformed]
        
        ax.scatter(plot_x, plot_y, c='red', s=50, marker='x', label='Ground Truth')

    ax.set_title('BEV Feature Heatmap with Ground Truth Annotations', fontsize=16)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    print(f"带标注的BEV热力图已保存至: {save_path}")