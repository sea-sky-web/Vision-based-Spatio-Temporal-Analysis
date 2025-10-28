import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from data.wildtrack_loader import WildtrackDataset, collate_fn
from models.model_wrapper import BEVNet
from utils.visualization import save_bev_heatmap
from typing import List, Dict


def load_cfg(path: str):
    # Ensure UTF-8 to support comments and non-ASCII characters in YAML
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg):
    opt_name = cfg['TRAIN']['OPT']
    lr = float(cfg['TRAIN']['LR'])
    wd = float(cfg['TRAIN']['WEIGHT_DECAY'])
    if opt_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg):
    name = cfg['TRAIN']['LR_SCHEDULER']
    if name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['TRAIN']['EPOCHS'])


def compute_metrics(pred_boxes, gt_targets, match_dist=0.5):
    # pred_boxes: List[Tensor[K,4]] in meters; gt_targets: List[Dict]
    tp, fp, fn = 0, 0, 0
    loc_errors = []
    for b in range(len(gt_targets)):
        gt = gt_targets[b].get('boxes_world', torch.zeros(0,4))
        preds = pred_boxes[b] if pred_boxes[b] is not None else torch.zeros(0,4)
        used = torch.zeros(gt.shape[0], dtype=torch.bool)
        for i in range(preds.shape[0]):
            p = preds[i, :2]
            if gt.shape[0] == 0:
                fp += 1
                continue
            dists = torch.norm(gt[:, :2] - p.unsqueeze(0), dim=1)
            min_dist, j = torch.min(dists, dim=0)
            if min_dist.item() <= match_dist and not used[j]:
                tp += 1
                used[j] = True
                loc_errors.append(min_dist.item())
            else:
                fp += 1
        fn += int((~used).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    mle = float(sum(loc_errors) / max(1, len(loc_errors)))
    return precision, recall, f1, mle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save_vis', action='store_true', default=False)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(cfg['RUNTIME']['DEVICE'] if torch.cuda.is_available() else 'cpu')

    ds = WildtrackDataset(cfg)
    # Wildtrack精简规范：固定切分 train/val=400/100（若总帧不足则回退随机切分）
    total = len(ds)
    if total >= 500:
        indices_train = list(range(0, 400))
        indices_val = list(range(400, 500))
        train_ds = torch.utils.data.Subset(ds, indices_train)
        val_ds = torch.utils.data.Subset(ds, indices_val)
    else:
        val_ratio = 0.2
        n_val = int(total * val_ratio)
        n_train = total - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val])

    dl_train = DataLoader(train_ds, batch_size=cfg['DATA']['BATCH_SIZE'], shuffle=True,
                          num_workers=cfg['RUNTIME']['NUM_WORKERS'], collate_fn=collate_fn)
    dl_val = DataLoader(val_ds, batch_size=cfg['DATA']['BATCH_SIZE'], shuffle=False,
                        num_workers=cfg['RUNTIME']['NUM_WORKERS'], collate_fn=collate_fn)

    model = BEVNet(cfg)
    model = model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    best_f1 = -1.0
    save_dir = os.path.join(os.path.dirname(args.config), '..', cfg['RUNTIME']['SAVE_DIR']).replace('\\', '/')
    os.makedirs(save_dir, exist_ok=True)

    def _summarize_batch_gt(targets: List[Dict]) -> str:
        counts = []
        for t in targets:
            n_boxes = int(t.get('boxes_world', torch.zeros(0,4)).shape[0]) if t.get('boxes_world', None) is not None else 0
            centers = t.get('centers_world', None)
            n_centers = int(centers.shape[0]) if centers is not None else 0
            counts.append(max(n_boxes, n_centers))
        return f"GT per-sample: {counts} | total={sum(counts)}"
    
    
    def _summarize_calib(calib: Dict) -> str:
        # calib['intrinsic'] and calib['extrinsic'] are List[List[Tensor]] of shape [B][V]
        if not isinstance(calib.get('intrinsic', None), list) or len(calib['intrinsic']) == 0:
            return "Rt angles(rad): [] | t_norms: []"
        K_views = calib['intrinsic'][0]
        Rt_views = calib['extrinsic'][0]
        angles = []
        tnorms = []
        for Rt in Rt_views:
            try:
                R = Rt[:3, :3]
                angle = float(torch.arccos(torch.clamp(((torch.trace(R) - 1.0) / 2.0), -1.0, 1.0)))
                tnorm = float(torch.norm(Rt[:3, 3]))
            except Exception:
                angle, tnorm = float('nan'), float('nan')
            angles.append(round(angle, 3))
            tnorms.append(round(tnorm, 3))
        return f"Rt angles(rad): {angles} | t_norms: {tnorms}"

    for epoch in range(cfg['TRAIN']['EPOCHS']):
        model.train()
        running_loss = 0.0
        first_batch_logged = False
        for batch in dl_train:
            batch['images'] = batch['images'].to(device)
            # move calib tensors to device
            for b in range(len(batch['calib']['intrinsic'])):
                batch['calib']['intrinsic'][b] = [k.to(device) for k in batch['calib']['intrinsic'][b]]
                batch['calib']['extrinsic'][b] = [e.to(device) for e in batch['calib']['extrinsic'][b]]
            if not first_batch_logged:
                print(f"[Train][Epoch {epoch}] { _summarize_batch_gt(batch['targets']) }")
                print(f"[Train][Epoch {epoch}] { _summarize_calib(batch['calib']) }")
                first_batch_logged = True
            preds = model(batch)
            loss_dict = model.loss(preds, batch['targets'], cfg['LOSS'])
            loss = loss_dict['total_loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            precisions, recalls, f1s, mles = [], [], [], []
            first_val_logged = False
            for batch in dl_val:
                batch['images'] = batch['images'].to(device)
                for b in range(len(batch['calib']['intrinsic'])):
                    batch['calib']['intrinsic'][b] = [k.to(device) for k in batch['calib']['intrinsic'][b]]
                    batch['calib']['extrinsic'][b] = [e.to(device) for e in batch['calib']['extrinsic'][b]]
                if not first_val_logged:
                    print(f"[Val][Epoch {epoch}] { _summarize_batch_gt(batch['targets']) }")
                    print(f"[Val][Epoch {epoch}] { _summarize_calib(batch['calib']) }")
                    first_val_logged = True
                preds = model(batch)
                # 评估使用thr=0.4, NMS=0.5m已在forward/decode中处理
                p, r, f1, mle = compute_metrics(preds['boxes'], batch['targets'], match_dist=cfg['EVAL']['NMS_DIST_M'])
                precisions.append(p); recalls.append(r); f1s.append(f1); mles.append(mle)
                if args.save_vis:
                    save_bev_heatmap(preds['heatmap'], os.path.join(cfg['RUNTIME']['OUTPUT_DIR'], f'epoch{epoch}_hm.png'))
            mean_p = sum(precisions)/max(1,len(precisions))
            mean_r = sum(recalls)/max(1,len(recalls))
            mean_f1 = sum(f1s)/max(1,len(f1s))
            mean_mle = sum(mles)/max(1,len(mles))
            print(f"Epoch {epoch}: P={mean_p:.3f} R={mean_r:.3f} F1={mean_f1:.3f} MLE={mean_mle:.3f} train_loss={running_loss/len(dl_train):.4f}")

            # checkpoint
            last_path = os.path.join(save_dir, 'last.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'f1': mean_f1}, last_path)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_path = os.path.join(save_dir, 'best.pth')
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'f1': mean_f1}, best_path)
                print(f"Saved new best checkpoint: {best_path}")


if __name__ == '__main__':
    main()