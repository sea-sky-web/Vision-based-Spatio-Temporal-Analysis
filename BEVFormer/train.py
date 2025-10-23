"""Standardized training framework for BEV Fusion (Wildtrack)

This script implements a robust training and validation pipeline:
- Dataset split into train/val via torch.utils.data.random_split
- Training loop with forward, loss, backward, optimizer step, and epoch average loss
- Validation loop with forward-only, GT generation, peak extraction, matching, and metrics aggregation
- Checkpointing: save best_model.pth based on validation F1
- Diagnostics: visualize only first batch per epoch or controlled by --visualize
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def main():
    # ---- Arguments ----
    parser = argparse.ArgumentParser(description="BEV Fusion Training Framework")
    parser.add_argument("--config", type=str, default="configs/wildtrack_v1.py", help="Config module path")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization during validation (first batch per epoch)",
    )
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except Exception as e:
        print("PyTorch is required. Please install dependencies first.")
        print("Error:", e)
        sys.exit(1)

    # ---- Imports (project modules) ----
    from configs import wildtrack_v1 as cfg
    from datasets.wildtrack_dataset import WildtrackDataset
    from models.bev_fusion_net import BEVFusionNet
    from models.modules.losses import FocalLoss
    from utils.heatmap_utils import build_heatmap_from_points, world_to_bev_indices
    from utils.bev_utils import (
        decode_position_id,
        analyze_annotation_bounds,
        update_config_with_bev_bounds,
        visualize_annotation_distribution,
    )
    from utils.nms_utils import simple_nms_peaks
    from utils.eval_utils import extract_peaks_from_heatmap
    from utils.visualization import visualize_feature_map

    # Default visualization from config unless overridden by CLI
    visualize = args.visualize or bool(getattr(cfg, "SAVE_VIS", False))

    print("Config:", cfg)

    # ---- Dataset ----
    ds = WildtrackDataset(cfg)

    # ---- Output directory ----
    output_dir = os.path.join(cfg.ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Auto analyze annotations & update BEV_BOUNDS ----
    try:
        if getattr(ds, "ann_dir", None) and ds.ann_dir and ds.ann_dir.exists():
            print(f"分析标注边界以推断BEV_BOUNDS: {ds.ann_dir}")
            bounds_info = analyze_annotation_bounds(str(ds.ann_dir), sample_size=100)
            if bounds_info is not None:
                update_ok = update_config_with_bev_bounds(cfg, bounds_info)
                if update_ok and visualize:
                    vis_path = os.path.join(output_dir, "bev_analysis.png")
                    visualize_annotation_distribution(str(ds.ann_dir), save_path=vis_path, sample_size=50)
            else:
                print("警告：标注边界分析未返回结果，继续使用默认BEV_BOUNDS")
        else:
            print("提示：未检测到有效标注目录，跳过BEV_BOUNDS自动推断")
    except Exception as e:
        print(f"BEV边界分析失败: {e}")

    # ---- Collate Function ----
    def custom_collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate that filters None samples and standardizes batch structure.

        - Filters out None entries produced by dataset for corrupted frames
        - Returns None when batch is empty to allow loader iteration to skip gracefully
        - Converts images to a stacked tensor of shape (B, NumCams, C, H, W)
        - Passes through meta fields; intrinsics/extrinsics remain as per-sample lists
        """
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        # Images
        images: List[torch.Tensor] = []
        for b in batch:
            img = b.get("images")
            if isinstance(img, torch.Tensor):
                images.append(img)
        try:
            images_stacked = torch.stack(images, dim=0)  # (B, NumCams, C, H, W)
        except Exception as e:
            print(f"堆叠图像失败: {e}")
            return None

        # Meta passthrough (lists kept per-sample)
        meta = {
            "frame_idx": [b.get("meta", {}).get("frame_idx", 0) for b in batch],
            "filename": [b.get("meta", {}).get("filename", "") for b in batch],
            "paths": [b.get("meta", {}).get("paths", []) for b in batch],
            "intrinsics": [b.get("meta", {}).get("intrinsics", []) for b in batch],
            "distortions": [b.get("meta", {}).get("distortions", []) for b in batch],
            "extrinsics": [b.get("meta", {}).get("extrinsics", []) for b in batch],
            # annotations reside at sample top-level
            "annotations": [b.get("annotations", []) for b in batch],
        }
        return {"images": images_stacked, "meta": meta}

    # ---- Dataset Split (80/20) ----
    # NOTE: 关键修改：使用random_split进行标准化的训练/验证划分
    total_len = len(ds)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(ds, [train_len, val_len], generator=generator)

    # ---- DataLoaders ----
    # 关键修改：train_loader启用shuffle=True；val_loader禁用shuffle
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=getattr(cfg, "BATCH_SIZE", 2),
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=getattr(cfg, "BATCH_SIZE", 2),
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    # ---- Model, Optimizer, Loss ----
    model = BEVFusionNet(cfg)
    optimizer = optim.Adam(
        model.parameters(),
        lr=getattr(cfg, "LR", 1e-3),
        weight_decay=getattr(cfg, "WEIGHT_DECAY", 0.0),
    )
    loss_type = getattr(cfg, "LOSS_TYPE", "focal")
    if loss_type == "focal":
        criterion = FocalLoss(
            alpha=getattr(cfg, "FOCAL_ALPHA", 0.25),
            gamma=getattr(cfg, "FOCAL_GAMMA", 2.0),
            reduction="mean",
        )
    else:
        criterion = torch.nn.MSELoss()

    # ---- Training & Validation Loops ----
    epochs = int(getattr(cfg, "EPOCHS", 1))
    best_f1_score = 0.0
    bev_bounds = getattr(cfg, "BEV_BOUNDS", (-6.0, 6.0, -2.0, 2.0))

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")

        # ----- Training Phase -----
        # 关键修改：标准训练流程（train()，前向/损失/反传/步进，记录平均损失）
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            images = batch["images"]
            meta = batch["meta"]

            # Convert intrinsics/extrinsics per-sample to torch tensors
            intrinsics_list: List[List[np.ndarray]] = meta.get("intrinsics", [])
            extrinsics_list: List[List[np.ndarray]] = meta.get("extrinsics", [])

            # For the model API, pass lists of per-sample tensors
            intrinsics_t: List[torch.Tensor] = []
            extrinsics_t: List[torch.Tensor] = []
            for s_idx in range(len(intrinsics_list)):
                cam_intr = intrinsics_list[s_idx]
                cam_extr = extrinsics_list[s_idx]
                # Stack per-sample camera matrices into tensors
                K = torch.tensor(np.stack(cam_intr, axis=0), dtype=torch.float32)
                E = torch.tensor(np.stack(cam_extr, axis=0), dtype=torch.float32)
                intrinsics_t.append(K)
                extrinsics_t.append(E)

            # Forward
            result = model(
                images,
                intrinsics=intrinsics_t,
                extrinsics=extrinsics_t,
                meta=meta,
                return_bev=True,
            )

            if isinstance(result, tuple):
                out, bev = result  # out: [B,1,H,W]
            else:
                out = result
                bev = None

            # Build GT heatmaps from annotations for loss
            bev_h = cfg.BEV_SIZE[1] if isinstance(cfg.BEV_SIZE, (tuple, list)) and len(cfg.BEV_SIZE) == 3 else cfg.BEV_SIZE[0]
            bev_w = cfg.BEV_SIZE[2] if isinstance(cfg.BEV_SIZE, (tuple, list)) and len(cfg.BEV_SIZE) == 3 else cfg.BEV_SIZE[1]

            gt_list = []
            for bi in range(len(meta["annotations"])):
                annos = meta["annotations"][bi] if isinstance(meta["annotations"][bi], list) else []
                world_points: List[Tuple[float, float]] = []
                for ann in annos:
                    if isinstance(ann, dict) and "positionID" in ann:
                        gx, gy = decode_position_id(ann["positionID"])  # grid coords
                        # Convert to meters (Wildtrack world grid to approx meters)
                        world_width = 12.0
                        world_height = 4.0
                        scale_x = world_width / 1440.0
                        scale_y = world_height / 480.0
                        wx = (gx * scale_x) - (world_width / 2.0)
                        wy = (gy * scale_y) - (world_height / 2.0)
                        world_points.append((wx, wy))
                if len(world_points) > 0:
                    pts = torch.tensor(world_points, dtype=torch.float32, device=out.device)
                    sigma = getattr(cfg, "HEATMAP_SIGMA", 2.0)
                    gt = build_heatmap_from_points(pts, bev_bounds, (bev_h, bev_w), sigma=sigma)  # [1,H,W]
                else:
                    gt = torch.zeros(1, bev_h, bev_w, dtype=torch.float32, device=out.device)
                gt_list.append(gt)

            gt_batch = torch.stack(gt_list, dim=0)  # [B,1,H,W]

            # Loss + backward + step
            loss = criterion(out, gt_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_batches += 1

        avg_train_loss = train_loss_sum / max(1, train_batches)
        print(f"Epoch {epoch + 1}: 平均训练损失 = {avg_train_loss:.6f}")

        # ----- Validation Phase -----
        # 关键修改：标准验证流程（eval()，no_grad，前向与指标累计）
        model.eval()
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_matches_dists: List[float] = []
        total_pred = 0
        total_gt = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch is None:
                    continue

                images = batch["images"]
                meta = batch["meta"]

                # Convert intrinsics/extrinsics per-sample to torch tensors
                intrinsics_list = meta.get("intrinsics", [])
                extrinsics_list = meta.get("extrinsics", [])
                intrinsics_t = []
                extrinsics_t = []
                for s_idx in range(len(intrinsics_list)):
                    K = torch.tensor(np.stack(intrinsics_list[s_idx], axis=0), dtype=torch.float32)
                    E = torch.tensor(np.stack(extrinsics_list[s_idx], axis=0), dtype=torch.float32)
                    intrinsics_t.append(K)
                    extrinsics_t.append(E)

                # Forward-only
                result = model(
                    images,
                    intrinsics=intrinsics_t,
                    extrinsics=extrinsics_t,
                    meta=meta,
                    return_bev=True,
                )
                if isinstance(result, tuple):
                    out, bev = result
                else:
                    out = result
                    bev = None

                # Metrics per batch
                bev_h = cfg.BEV_SIZE[1] if isinstance(cfg.BEV_SIZE, (tuple, list)) and len(cfg.BEV_SIZE) == 3 else cfg.BEV_SIZE[0]
                bev_w = cfg.BEV_SIZE[2] if isinstance(cfg.BEV_SIZE, (tuple, list)) and len(cfg.BEV_SIZE) == 3 else cfg.BEV_SIZE[1]

                # For each sample in batch
                for b in range(out.shape[0]):
                    out_map = torch.sigmoid(out[b, 0].detach())
                    if getattr(cfg, "APPLY_NMS", False):
                        out_map = simple_nms_peaks(out_map, kernel_size=getattr(cfg, "NMS_KERNEL", 3))

                    peaks = extract_peaks_from_heatmap(
                        out_map,
                        threshold=getattr(cfg, "PEAK_THRESHOLD", None),
                        kernel_size=getattr(cfg, "NMS_KERNEL", 3),
                    )
                    pred_idxs = (
                        torch.tensor([[p[0], p[1]] for p in peaks], dtype=torch.long, device=out.device)
                        if len(peaks) > 0
                        else torch.empty(0, 2, dtype=torch.long, device=out.device)
                    )

                    # Build GT world points
                    annos = meta["annotations"][b] if b < len(meta["annotations"]) else []
                    world_points: List[Tuple[float, float]] = []
                    for ann in annos:
                        if isinstance(ann, dict) and "positionID" in ann:
                            gx, gy = decode_position_id(ann["positionID"])  # grid coords
                            world_width = 12.0
                            world_height = 4.0
                            scale_x = world_width / 1440.0
                            scale_y = world_height / 480.0
                            wx = (gx * scale_x) - (world_width / 2.0)
                            wy = (gy * scale_y) - (world_height / 2.0)
                            world_points.append((wx, wy))

                    if len(world_points) > 0:
                        pts_xy = torch.tensor(world_points, dtype=torch.float32, device=out.device)
                        gt_idxs_all, mask = world_to_bev_indices(pts_xy, bev_bounds, (bev_h, bev_w))
                        gt_idxs = gt_idxs_all[mask]
                    else:
                        gt_idxs = torch.empty(0, 2, dtype=torch.long, device=out.device)

                    # Greedy matching in meters
                    min_x, max_x, min_y, max_y = bev_bounds
                    res_x = (max_x - min_x) / float(bev_w)
                    res_y = (max_y - min_y) / float(bev_h)
                    Mp = pred_idxs.shape[0]
                    Mg = gt_idxs.shape[0]

                    if Mp == 0 and Mg == 0:
                        tp = 0
                        fp = 0
                        fn = 0
                        match_d = []
                    elif Mp == 0:
                        tp = 0
                        fp = 0
                        fn = Mg
                        match_d = []
                    elif Mg == 0:
                        tp = 0
                        fp = Mp
                        fn = 0
                        match_d = []
                    else:
                        dif = pred_idxs.float().unsqueeze(1) - gt_idxs.float().unsqueeze(0)  # [Mp, Mg, 2] (y,x)
                        dx_m = dif[..., 1] * res_x
                        dy_m = dif[..., 0] * res_y
                        dists = torch.sqrt(dx_m * dx_m + dy_m * dy_m)  # [Mp, Mg]
                        # Greedy matching
                        pairs: List[Tuple[int, int, float]] = []
                        for i in range(Mp):
                            for j in range(Mg):
                                pairs.append((i, j, float(dists[i, j].item())))
                        pairs.sort(key=lambda x: x[2])
                        thresh = float(getattr(cfg, "MATCH_DIST_THRESH", 0.5))
                        matched_p = set()
                        matched_g = set()
                        match_d = []
                        for i, j, d in pairs:
                            if d > thresh:
                                break
                            if i in matched_p or j in matched_g:
                                continue
                            matched_p.add(i)
                            matched_g.add(j)
                            match_d.append(d)
                        tp = len(matched_p)
                        fp = Mp - tp
                        fn = Mg - tp

                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total_pred += Mp
                    total_gt += Mg
                    total_matches_dists.extend(match_d)

                # Visualization: only first val batch per epoch if enabled
                if visualize and batch_idx == 0 and bev is not None:
                    for b in range(min(bev.shape[0], 1)):
                        bev_vis_path = os.path.join(output_dir, f"bev_features_epoch{epoch+1}_valbatch{batch_idx}_sample{b}.png")
                        visualize_feature_map(bev[b], bev_vis_path, title=f"BEV特征图 - Epoch{epoch+1} ValBatch{batch_idx} Sample{b}")

                        out_vis_path = os.path.join(output_dir, f"output_heatmap_epoch{epoch+1}_valbatch{batch_idx}_sample{b}.png")
                        out_map_np = torch.sigmoid(out[b, 0].detach()).cpu().numpy()
                        plt.figure(figsize=(10, 8))
                        plt.imshow(out_map_np, cmap="jet")
                        plt.colorbar(label="热图值")
                        plt.title(f"BEV输出热图 - Epoch{epoch+1} ValBatch{batch_idx} Sample{b}")
                        plt.xlabel("X")
                        plt.ylabel("Y")
                        plt.savefig(out_vis_path)
                        plt.close()

        # Aggregate metrics
        precision = total_tp / max(1, total_tp + total_fp)
        recall = total_tp / max(1, total_tp + total_fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        mean_distance = float(sum(total_matches_dists) / max(1, len(total_matches_dists)))
        print(
            f"验证指标: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, 平均匹配距离={mean_distance:.3f}m, 预测数={total_pred}, GT数={total_gt}"
        )

        # ----- Checkpointing -----
        # 关键修改：基于最佳F1保存模型参数
        if f1 > best_f1_score:
            best_f1_score = f1
            ckpt_path = os.path.join(output_dir, "best_model.pth")
            try:
                torch.save(model.state_dict(), ckpt_path)
                print(f"保存最佳模型到 {ckpt_path} (F1={best_f1_score:.3f})")
            except Exception as e:
                print(f"保存模型失败: {e}")

    print("\n训练与验证完成！")
    print(f"所有输出已保存到 {output_dir}")


if __name__ == "__main__":
    main()
