import os
import torch
import numpy as np
import sys
from mmdet3d.apis import init_model
from mmcv.parallel import collate, scatter

# 添加自定义模块路径
sys.path.append(os.path.abspath('.'))

from custom.wildtrack_dataset import CustomWildTrackDataset
from custom.visualize_bev import visualize_bev_with_annotations

def main():
    """验证BEVFormer在WildTrack数据集上的性能并生成带标注的可视化"""
    print("开始验证BEVFormer在WildTrack数据集上的性能...")
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 配置文件和检查点路径
    config_file = 'projects/configs/bevformer/custom_wildtrack.py'
    checkpoint_file = 'bevformer_base_epoch_24.pth'
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        print(f"错误: 配置文件 '{config_file}' 不存在。请确保路径正确。")
        return
    
    if not os.path.exists(checkpoint_file):
        print(f"错误: 检查点文件 '{checkpoint_file}' 不存在。")
        print("请从 https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_base_epoch_24.pth 下载并放置在 'BEVFormer/' 目录下。")
        return
    
    try:
        # 初始化模型
        print("正在加载模型...")
        model = init_model(config_file, checkpoint_file, device=device)
        model.eval()
        print("模型加载成功")
        
        # 创建数据集实例
        print("正在加载WildTrack数据集...")
        dataset = CustomWildTrackDataset(data_root='data/Wildtrack/')
        
        # 准备测试数据（第一帧）
        frame_id = 0  # 使用第一帧进行测试
        print(f"正在准备第 {frame_id} 帧的测试数据...")
        data = dataset.prepare_test_data(frame_id)

        # 转换数据格式以适应模型输入
        img_metas = [data['img_metas']]
        img = torch.from_numpy(data['img']).permute(0, 3, 1, 2).float().unsqueeze(0) # 增加batch维度
        
        # 提取BEV特征
        print("正在提取真实的BEV特征...")
        with torch.no_grad():
            # BEVFormer的正确推理流程
            # 我们需要调用 forward 并从中提取 bev_embed
            result = model.forward(img=[img], img_metas=[img_metas], return_loss=False)
            
            # BEV特征通常存储在 'bev_embed' 键中
            # 它的形状是 [1, H*W, C] 或 [1, C, H, W]，需要根据模型调整
            if 'bev_embed' in result:
                bev_embed = result['bev_embed']

                # 如果是 [1, H*W, C] 格式, 转换为 [1, C, H, W]
                if bev_embed.dim() == 3:
                    bev_h = model.pts_bbox_head.bev_h
                    bev_w = model.pts_bbox_head.bev_w
                    bev_embed = bev_embed.permute(0, 2, 1).view(1, -1, bev_h, bev_w)
            else:
                raise ValueError("无法在模型输出中找到 'bev_embed'。请检查模型结构或输出。")

        # 可视化BEV特征并叠加标注
        print("正在可视化BEV特征并叠加3D标注...")
        
        # 定义BEV空间范围和尺寸，用于坐标转换
        # 这些值应与配置文件中的 'pc_range' 和 'bev_h'/'bev_w' 匹配
        pc_range = model.pts_bbox_head.pc_range  # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        bev_height = model.pts_bbox_head.bev_h
        bev_width = model.pts_bbox_head.bev_w

        visualize_bev_with_annotations(
            bev_embed,
            data['annotations'],
            pc_range=pc_range,
            bev_height=bev_height,
            bev_width=bev_width,
            save_path='test_bev_with_annotations.png'
        )
        
        print("\n验证完成！")
        print("请查看 'test_bev_with_annotations.png' 文件以检查带标注的BEV热力图。")
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("这通常意味着数据文件或符号链接不正确。请仔细检查 '数据准备' 步骤。")
    except Exception as e:
        print(f"\n验证过程中出现严重错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()