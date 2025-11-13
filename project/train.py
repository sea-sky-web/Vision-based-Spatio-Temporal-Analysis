import os
import argparse
import yaml
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

from data.wildtrack_loader import WildtrackDataset, collate_fn
from models.model_wrapper import BEVNet
from utils.visualization import save_bev_heatmap
from typing import List, Dict
import matplotlib.pyplot as plt


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
    if name == 'cosine_warm':
        warm = int(cfg['TRAIN'].get('WARMUP_EPOCHS', 3))
        total = int(cfg['TRAIN']['EPOCHS'])
        def lr_lambda(epoch):
            if epoch < warm:
                return float(epoch + 1) / float(max(1, warm))
            return 1.0
        base = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total - warm))
        class _Seq:
            def __init__(self, scheds):
                self.scheds = scheds
            def step(self):
                for s in self.scheds:
                    s.step()
        return _Seq([base, cos])
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
    return precision, recall, f1, mle, tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save_vis', action='store_true', default=False)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(cfg['RUNTIME']['DEVICE'] if torch.cuda.is_available() else 'cpu')
    # Enable cuDNN autotune and improved matmul precision when on CUDA
    if device.type == 'cuda':
        cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass
        print(f"[Runtime] Using CUDA: {torch.cuda.get_device_name(0)} | drivers CUDA {torch.version.cuda}")

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

    # DataLoader tweaks for GPU: pin_memory + persistent_workers when num_workers>0
    nw = int(cfg['RUNTIME']['NUM_WORKERS'])
    pin_mem = device.type == 'cuda'
    dl_train = DataLoader(
        train_ds,
        batch_size=cfg['DATA']['BATCH_SIZE'],
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    dl_val = DataLoader(
        val_ds,
        batch_size=cfg['DATA']['BATCH_SIZE'],
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )

    model = BEVNet(cfg)
    model = model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Mixed precision (AMP) for faster training on GPU
    use_amp = bool(cfg.get('RUNTIME', {}).get('USE_AMP', True)) and device.type == 'cuda'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        scaler = GradScaler(enabled=use_amp)

    best_f1 = -1.0
    save_dir = os.path.join(os.path.dirname(args.config), '..', cfg['RUNTIME']['SAVE_DIR']).replace('\\', '/')
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tb'))
    interval = int(cfg.get('EVAL', {}).get('INTERVAL', 1))
    mem_limit = int(cfg.get('RUNTIME', {}).get('MEMORY_LIMIT_PERCENT', 90))
    baseline_name = str(cfg.get('EVAL', {}).get('BASELINE_MODEL', 'baseline'))
    baseline_f1 = float(cfg.get('EVAL', {}).get('BASELINE_F1', 0.0))
    improve_thr = float(cfg.get('EVAL', {}).get('IMPROVEMENT_THRESHOLD', 5.0))
    early_patience = int(cfg.get('TRAIN', {}).get('PATIENCE', 0))

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

    accum_steps = int(cfg['TRAIN'].get('ACCUM_STEPS', 1))
    debug_max = int(cfg.get('RUNTIME', {}).get('DEBUG_MAX_STEPS', 0))
    global_step = 0
    no_improve_epochs = 0
    train_loss_curve = []
    val_f1_curve = []
    for epoch in range(cfg['TRAIN']['EPOCHS']):
        model.train()
        running_loss = 0.0
        first_batch_logged = False
        step_count = 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        for batch in dl_train:
            batch['images'] = batch['images'].to(device, non_blocking=True)
            # move calib tensors to device
            for b in range(len(batch['calib']['intrinsic'])):
                batch['calib']['intrinsic'][b] = [k.to(device) for k in batch['calib']['intrinsic'][b]]
                batch['calib']['extrinsic'][b] = [e.to(device) for e in batch['calib']['extrinsic'][b]]
            if not first_batch_logged:
                print(f"[Train][Epoch {epoch}] { _summarize_batch_gt(batch['targets']) }")
                print(f"[Train][Epoch {epoch}] { _summarize_calib(batch['calib']) }")
                first_batch_logged = True
            if use_amp:
                with autocast(dtype=torch.float16):
                    preds = model(batch)
                    loss_dict = model.loss(preds, batch['targets'], cfg['LOSS'])
                    loss = loss_dict['total_loss'] / float(max(1, accum_steps))
                scaler.scale(loss).backward()
                if (step_count + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                preds = model(batch)
                loss_dict = model.loss(preds, batch['targets'], cfg['LOSS'])
                loss = loss_dict['total_loss'] / float(max(1, accum_steps))
                loss.backward()
                if (step_count + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item()
            step_count += 1
            global_step += 1
            if step_count % 10 == 0:
                dt = time.perf_counter() - t0
                spp = step_count / max(1e-6, dt)
                print(f"[Train][Epoch {epoch}] steps={step_count} avg_steps/s={spp:.2f}")
            writer.add_scalar('train/loss_iter', float(loss.item()), global_step)
            if debug_max > 0 and step_count >= debug_max:
                break
        scheduler.step()

        # validation
        model.eval()
        do_eval = ((epoch + 1) % max(1, interval) == 0)
        with torch.no_grad():
            precisions, recalls, f1s, mles = [], [], [], []
            tps, fps, fns = 0, 0, 0
            first_val_logged = False
            val_step_count = 0
            for batch in dl_val if do_eval else []:
                batch['images'] = batch['images'].to(device, non_blocking=True)
                for b in range(len(batch['calib']['intrinsic'])):
                    batch['calib']['intrinsic'][b] = [k.to(device) for k in batch['calib']['intrinsic'][b]]
                    batch['calib']['extrinsic'][b] = [e.to(device) for e in batch['calib']['extrinsic'][b]]
                if not first_val_logged:
                    print(f"[Val][Epoch {epoch}] { _summarize_batch_gt(batch['targets']) }")
                    print(f"[Val][Epoch {epoch}] { _summarize_calib(batch['calib']) }")
                    first_val_logged = True
                if use_amp:
                    with autocast(dtype=torch.float16):
                        preds = model(batch)
                else:
                    preds = model(batch)
                # 评估使用thr=0.4, NMS=0.5m已在forward/decode中处理
                p, r, f1, mle, tp, fp, fn = compute_metrics(preds['boxes'], batch['targets'], match_dist=cfg['EVAL']['NMS_DIST_M'])
                precisions.append(p); recalls.append(r); f1s.append(f1); mles.append(mle)
                tps += tp; fps += fp; fns += fn
                if args.save_vis:
                    save_bev_heatmap(preds['heatmap'], os.path.join(cfg['RUNTIME']['OUTPUT_DIR'], f'epoch{epoch}_hm.png'))
                val_step_count += 1
                if debug_max > 0 and val_step_count >= debug_max:
                    break
            mean_p = sum(precisions)/max(1,len(precisions))
            mean_r = sum(recalls)/max(1,len(recalls))
            mean_f1 = sum(f1s)/max(1,len(f1s))
            mean_mle = sum(mles)/max(1,len(mles))
            train_loss_epoch = running_loss/len(dl_train)
            train_loss_curve.append(float(train_loss_epoch))
            if do_eval:
                val_f1_curve.append(float(mean_f1))
            stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            phase = 'eval' if do_eval else 'train'
            print(f"[{stamp}] phase={phase} epoch={epoch} loss={train_loss_epoch:.4f} P={mean_p:.3f} R={mean_r:.3f} F1={mean_f1:.3f} MLE={mean_mle:.3f} TP={tps} FP={fps} FN={fns}")
            if _HAS_NVML:
                try:
                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    mem_percent = int(mem.used * 100 / max(1, mem.total))
                    print(f"[GPU] util={util.gpu}% mem_used={mem.used/1024/1024:.1f}MB mem%={mem_percent}%")
                    if mem_percent >= mem_limit:
                        trig = os.path.join(save_dir, 'mem_triggered.pth')
                        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'f1': mean_f1}, trig)
                        print(f"Saved memory-triggered checkpoint: {trig}")
                except Exception:
                    pass
            if _HAS_PSUTIL:
                try:
                    cpu = psutil.cpu_percent(interval=None)
                    ram = psutil.virtual_memory().percent
                    print(f"[SYS] cpu={cpu}% ram={ram}%")
                except Exception:
                    pass
            writer.add_scalar('val/precision', float(mean_p), epoch)
            writer.add_scalar('val/recall', float(mean_r), epoch)
            writer.add_scalar('val/f1', float(mean_f1), epoch)
            writer.add_scalar('val/mle', float(mean_mle), epoch)

            # checkpoint
            last_path = os.path.join(save_dir, 'last.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'f1': mean_f1}, last_path)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_path = os.path.join(save_dir, 'best.pth')
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'f1': mean_f1}, best_path)
                print(f"Saved new best checkpoint: {best_path}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            if early_patience > 0 and no_improve_epochs >= early_patience and do_eval:
                print(f"Early stopping at epoch {epoch} due to no improvement")
                break

    try:
        plt.figure(figsize=(6,4))
        plt.plot(train_loss_curve, label='train_loss')
        if len(val_f1_curve) > 0:
            plt.plot(val_f1_curve, label='val_f1')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
        plt.close()
    except Exception:
        pass
    try:
        writer.close()
    except Exception:
        pass
if __name__ == '__main__':
    main()