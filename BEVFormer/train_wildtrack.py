import os
import torch
import mmcv
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                       build_optimizer, build_runner)
from mmcv.utils import get_git_hash
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.utils import collect_env
import time

def main():
    """主训练函数"""
    # 1. 加载配置
    cfg = mmcv.Config.fromfile('projects/configs/bevformer/custom_wildtrack.py')

    # 2. 设置工作目录和分布式训练
    cfg.work_dir = './work_dirs/bevformer_wildtrack_finetune'
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)

    # 初始化分布式训练（如果需要）
    # 这里简化为单GPU训练
    distributed = False

    # 3. 创建日志记录器
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = mmcv.get_logger(name='mmdet3d', log_file=log_file, log_level=cfg.log_level)

    # 4. 设置随机种子
    set_random_seed(cfg.seed, deterministic=False)

    # 5. 构建数据集
    # 注意：需要修改 wildtrack_dataset.py 以适应 mmdet 的数据集格式
    datasets = [build_dataset(cfg.data.train)]

    # 6. 构建模型
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights() # 初始化权重

    # 7. 加载预训练权重
    # 我们只加载backbone和transformer的权重，不加载分类头的权重
    checkpoint = load_checkpoint(model, 'bevformer_base_epoch_24.pth', map_location='cpu', strict=False, logger=logger)
    logger.info("成功加载预训练权重（除分类头外）")

    # 8. 构建运行器 (Runner)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=dict(
                env_info=collect_env(),
                git_hash=get_git_hash(),
                seed=cfg.seed,
                config=cfg.pretty_text
            )
        ))

    # 9. 开始训练
    runner.run([build_dataloader(ds, **cfg.data.dataloader_cfg) for ds in datasets], cfg.workflow)

if __name__ == '__main__':
    main()
