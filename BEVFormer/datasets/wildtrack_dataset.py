from pathlib import Path
import json
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import xml.etree.ElementTree as ET


# NOTE: 本文件已重构为仅支持 Wildtrack 标准 XML 标定格式。
# - 完全移除 calibrations.json 解析逻辑及其相关回退
# - 移除数据集缺失时生成随机/合成数据的逻辑
# - 当路径或标定目录错误时，直接抛出 FileNotFoundError
# - 保留 transform 预处理、运行时容错（损坏图像时跳过样本）、以及外参方向性校验


def _parse_intrinsic_xml(xml_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """解析内参 XML（OpenCV 格式）为 numpy 数组。

    期望结构包含 `camera_matrix/data`（9个数，3x3）以及可选的
    `distortion_coefficients/data`（1D数组）。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        K = None
        dist = None

        for child in root:
            if child.tag == "camera_matrix":
                data_elem = child.find("data")
                if data_elem is not None and data_elem.text:
                    vals = [float(x) for x in data_elem.text.split()]
                    if len(vals) >= 9:
                        K = np.array(vals[:9], dtype=np.float32).reshape(3, 3)
            elif child.tag == "distortion_coefficients":
                data_elem = child.find("data")
                if data_elem is not None and data_elem.text:
                    dist = np.array([float(x) for x in data_elem.text.split()], dtype=np.float32)

        return K, dist
    except Exception as e:
        print(f"警告：解析内参文件 {xml_path} 失败: {e}")
        return None, None


def _parse_extrinsic_xml(xml_path: str) -> Optional[np.ndarray]:
    """解析外参 XML 并返回 4x4 姿态矩阵（支持 rvec+tvec）。

    - 优先使用 OpenCV Rodrigues；若不可用则使用数学公式回退。
    - 返回值为世界->相机（world->camera）的 4x4 齐次矩阵。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        rvec = None
        tvec = None

        for child in root:
            if child.tag == "rvec" and child.text:
                rvec_vals = [float(x) for x in child.text.split()]
                rvec = np.array(rvec_vals, dtype=np.float32)
            elif child.tag == "tvec" and child.text:
                tvec_vals = [float(x) for x in child.text.split()]
                tvec = np.array(tvec_vals, dtype=np.float32)

        if rvec is not None and tvec is not None:
            try:
                import cv2
                R, _ = cv2.Rodrigues(rvec.astype(np.float32))
            except Exception:
                # Rodrigues 回退实现
                theta = float(np.linalg.norm(rvec))
                if theta > 1e-6:
                    k = rvec / theta
                    Kmat = np.array(
                        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
                        dtype=np.float32,
                    )
                    R = (
                        np.eye(3, dtype=np.float32)
                        + np.sin(theta) * Kmat
                        + (1 - np.cos(theta)) * (Kmat @ Kmat)
                    )
                else:
                    R = np.eye(3, dtype=np.float32)

            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = tvec
            return extrinsic

        return None
    except Exception as e:
        print(f"警告：解析外参文件 {xml_path} 失败: {e}")
        return None


class WildtrackDataset(Dataset):
    """Wildtrack 多视角数据集加载器（XML 标定版本）。

    返回每个样本字典：
    - 'images': torch.FloatTensor，形状 (NumCams, C, H, W)，范围 [0,1]
    - 'meta': { 'frame_idx', 'filename', 'paths', 'intrinsics', 'distortions', 'extrinsics' }
    - 'annotations': list，来自该帧的注释（若存在）

    重构要点：
    - 仅通过 XML 解析相机内外参
    - 当路径错误时抛出 FileNotFoundError，不再生成合成数据
    - 在 __getitem__ 中保留 try-except：单张图损坏时记录警告并跳过该样本（返回 None）
    - 保留并在 __init__ 结尾调用外参方向性校验方法
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_root = Path(cfg.DATA_ROOT)
        self.num_cameras = int(getattr(cfg, "NUM_CAMERAS", 7))
        self.img_size = tuple(getattr(cfg, "IMAGE_SIZE", (3, 256, 256)))

        # 保留核心图像预处理流程：Resize + ToTensor
        self.transform = T.Compose(
            [
                T.Resize((self.img_size[1], self.img_size[2])),
                T.ToTensor(),
            ]
        )

        # —— 路径与目录严格检查 ——
        if not self.data_root.exists():
            raise FileNotFoundError(f"未找到数据集根目录: {self.data_root}")

        # 相机图像目录（C1..C{num_cameras}）
        self.cam_dirs: List[Path] = []
        for i in range(1, self.num_cameras + 1):
            d = self.data_root / "Image_subsets" / f"C{i}"
            if not d.exists():
                raise FileNotFoundError(f"未找到相机文件夹: {d}")
            self.cam_dirs.append(d)

        # 按相机 C1 的文件名作为帧列表（默认所有相机对齐同名）
        self.frame_files = sorted([p.name for p in self.cam_dirs[0].iterdir() if p.is_file()])
        if len(self.frame_files) == 0:
            raise FileNotFoundError(
                f"在 {self.cam_dirs[0]} 未发现任何图像文件，请检查数据集完整性"
            )
        self.length = len(self.frame_files)

        # —— 仅从 XML 解析相机内外参 ——
        cal_dir = self.data_root / "calibrations"
        intr_dir = cal_dir / "intrinsic_zero"  # 与 Wildtrack 去畸变图像一致的内参目录
        extr_dir = cal_dir / "extrinsic"
        if not intr_dir.exists() or not extr_dir.exists():
            raise FileNotFoundError(
                f"标定目录缺失：intrinsic_zero={intr_dir.exists()}, extrinsic={extr_dir.exists()}"
            )

        self.intrinsics: List[np.ndarray] = []
        self.distortions: List[np.ndarray] = []
        self.extrinsics: List[np.ndarray] = []

        # 为每个相机匹配对应的内参/外参 XML 文件（名称包含相机索引）
        intr_files = list(intr_dir.iterdir())
        extr_files = list(extr_dir.iterdir())
        for i in range(self.num_cameras):
            idx_str = str(i + 1)
            intr_path = next((p for p in intr_files if p.is_file() and idx_str in p.name), None)
            extr_path = next((p for p in extr_files if p.is_file() and idx_str in p.name), None)
            if intr_path is None:
                raise FileNotFoundError(f"相机{i+1} 未找到内参 XML 文件于 {intr_dir}")
            if extr_path is None:
                raise FileNotFoundError(f"相机{i+1} 未找到外参 XML 文件于 {extr_dir}")

            K, dist = _parse_intrinsic_xml(str(intr_path))
            Rt = _parse_extrinsic_xml(str(extr_path))
            if K is None:
                raise ValueError(f"解析内参失败：{intr_path}")
            if Rt is None:
                raise ValueError(f"解析外参失败：{extr_path}")
            if dist is None:
                # 畸变可选；若无则使用零向量（非合成图像，仅标定缺省）
                dist = np.zeros(4, dtype=np.float32)

            self.intrinsics.append(K.astype(np.float32))
            self.distortions.append(dist.astype(np.float32))
            self.extrinsics.append(Rt.astype(np.float32))

        # 注释位置（每帧 JSON）路径（存在即可使用）
        self.ann_dir = self.data_root / "annotations_positions"

        # —— 按 Resize 目标尺寸缩放内参（保持几何一致性）——
        try:
            target_h, target_w = self.img_size[1], self.img_size[2]
            # Wildtrack 去畸变图像的原始标定尺寸通常为 (1080, 1920)
            orig_h, orig_w = 1080, 1920
            if (target_h, target_w) != (orig_h, orig_w):
                scale_h = float(target_h) / float(orig_h)
                scale_w = float(target_w) / float(orig_w)
                scaled_intrinsics: List[np.ndarray] = []
                for K in self.intrinsics:
                    K_scaled = K.astype(np.float32).copy()
                    K_scaled[0, 0] *= scale_w
                    K_scaled[1, 1] *= scale_h
                    K_scaled[0, 2] *= scale_w
                    K_scaled[1, 2] *= scale_h
                    scaled_intrinsics.append(K_scaled)
                self.intrinsics = scaled_intrinsics
        except Exception as e:
            print(f"警告：缩放内参失败，继续使用原始内参。错误: {e}")

        # —— 校验并自动修正外参方向（必要时取逆）——
        try:
            self._verify_and_fix_extrinsics_orientation()
        except Exception as e:
            print(f"警告：外参方向校验失败，继续使用原始外参。错误: {e}")

    def __len__(self) -> int:
        return self.length

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        """按索引返回一个样本；若该样本存在损坏文件则返回 None（跳过）。

        保留运行时容错：
        - 单张图像损坏/不可读取：记录警告并跳过该样本（返回 None）
        - 确保 transform 被应用，返回标准化的 PyTorch 张量
        """
        C, H, W = self.img_size

        try:
            fname = self.frame_files[idx]
            images: List[torch.Tensor] = []
            paths: List[str] = []

            for cam_i, d in enumerate(self.cam_dirs):
                p = d / fname
                if not p.exists():
                    # 支持 png/jpg 扩展名回退
                    p_png = p.with_suffix(".png")
                    p_jpg = p.with_suffix(".jpg")
                    if p_png.exists():
                        p = p_png
                    elif p_jpg.exists():
                        p = p_jpg
                    else:
                        print(f"警告：相机{cam_i+1} 缺少图像文件 {fname} 于 {d}，跳过样本 {fname}")
                        return None  # 跳过整帧

                try:
                    img_tensor = self._load_image(p)
                except Exception as e:
                    print(f"警告：读取图像失败 {p}: {e}，跳过样本 {fname}")
                    return None  # 跳过整帧

                if not isinstance(img_tensor, torch.Tensor):
                    print(f"警告：transform 未返回张量 {p}，跳过样本 {fname}")
                    return None

                # 期望形状 (C, H, W)
                if img_tensor.ndim != 3 or img_tensor.shape[0] != C:
                    print(
                        f"警告：图像尺寸不匹配 {p}，期望C={C}，得到 {tuple(img_tensor.shape)}；跳过样本 {fname}"
                    )
                    return None

                images.append(img_tensor)
                paths.append(str(p))

            # 堆叠成 (NumCams, C, H, W)
            try:
                imgs = torch.stack(images, dim=0)
            except Exception as e:
                print(f"警告：堆叠多视角图像失败（{fname}）: {e}，跳过样本")
                return None

            meta = {
                "frame_idx": idx,
                "filename": fname,
                "paths": paths,
                # 注意：保持 numpy 数组；后续训练流程会统一处理/转换
                "intrinsics": self.intrinsics,
                "distortions": self.distortions,
                "extrinsics": self.extrinsics,
            }

            # 加载注释（若存在）
            ann = []
            try:
                if hasattr(self, "ann_dir") and self.ann_dir:
                    ann_path = self.ann_dir / fname.replace(".png", ".json").replace(".jpg", ".json")
                    if ann_path.exists():
                        with open(ann_path, "r", encoding="utf-8") as f:
                            ann = json.load(f)
            except Exception as e:
                print(f"警告：加载注释失败 {fname}: {e}")
                ann = []

            sample = {"images": imgs, "meta": meta, "annotations": ann}
            return sample

        except Exception as e:
            # 顶层防护：任何未预料的异常不阻断训练，直接跳过样本
            print(f"警告：处理样本 {idx} 失败: {e}，跳过该样本")
            return None

    def _verify_and_fix_extrinsics_orientation(self, threshold: float = 0.05) -> None:
        """检查每个相机外参是否为世界->相机方向，并在必要时自动取逆。

        方法：
        - 在地面平面 (Z=0) 采样网格点（依据配置的 BEV_BOUNDS，若未提供则使用默认范围）
        - 使用 H = K [r1, r2, t] 将地面点投影到图像平面
        - 统计落在图像范围内的比例 (in-bounds ratio)
        - 若比例过低（<threshold），尝试对外参求逆后再次评估；若改善明显则采用逆矩阵
        """
        if not self.intrinsics or not self.extrinsics:
            return

        H_img, W_img = self.img_size[1], self.img_size[2]
        try:
            bev_bounds = getattr(self.cfg, "BEV_BOUNDS", (-6.0, 6.0, -2.0, 2.0))
        except Exception:
            bev_bounds = (-6.0, 6.0, -2.0, 2.0)

        x_min, x_max, y_min, y_max = bev_bounds
        bev_h, bev_w = 64, 64
        xs = np.linspace(x_min, x_max, bev_w, dtype=np.float32)
        ys = np.linspace(y_min, y_max, bev_h, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs)  # (H, W)
        ones = np.ones_like(xx, dtype=np.float32)
        ground = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3).T  # [3, H*W]

        fixed_count = 0
        for idx, (K_np, E_np) in enumerate(zip(self.intrinsics, self.extrinsics)):
            try:
                if K_np is None or E_np is None:
                    continue
                K = K_np.astype(np.float32)
                E = E_np.astype(np.float32)

                R = E[:3, :3]
                t = E[:3, 3:4]
                G = np.concatenate([R[:, 0:1], R[:, 1:2], t], axis=1)  # [3,3]
                H_w2i = K @ G  # [3,3]

                uvw = H_w2i @ ground  # [3, H*W]
                w = uvw[2:3, :]
                w_safe = np.where(np.abs(w) < 1e-6, 1.0, w)
                u = (uvw[0:1, :] / w_safe).squeeze(0)
                v = (uvw[1:2, :] / w_safe).squeeze(0)
                in_image = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
                ratio = float(np.mean(in_image))

                if ratio < threshold:
                    try:
                        E_inv = np.linalg.inv(E)
                    except Exception:
                        E_inv = None
                    if E_inv is not None:
                        R2 = E_inv[:3, :3]
                        t2 = E_inv[:3, 3:4]
                        G2 = np.concatenate([R2[:, 0:1], R2[:, 1:2], t2], axis=1)
                        H2 = K @ G2
                        uvw2 = H2 @ ground
                        w2 = uvw2[2:3, :]
                        w2_safe = np.where(np.abs(w2) < 1e-6, 1.0, w2)
                        u2 = (uvw2[0:1, :] / w2_safe).squeeze(0)
                        v2 = (uvw2[1:2, :] / w2_safe).squeeze(0)
                        in_image2 = (u2 >= 0) & (u2 < W_img) & (v2 >= 0) & (v2 < H_img)
                        ratio2 = float(np.mean(in_image2))

                        if ratio2 > ratio + 0.10:
                            print(
                                f"外参方向修正：相机{idx+1} in-bounds {ratio:.3f} -> {ratio2:.3f}，改用外参逆矩阵"
                            )
                            self.extrinsics[idx] = E_inv.astype(np.float32)
                            fixed_count += 1
                        else:
                            print(
                                f"外参方向未修正：相机{idx+1} in-bounds {ratio:.3f}，逆矩阵提升不明显 ({ratio2:.3f})"
                            )
                    else:
                        print(f"警告：相机{idx+1}外参不可逆，跳过方向修正")
                else:
                    print(f"外参方向正常：相机{idx+1} in-bounds 比例 {ratio:.3f}")
            except Exception as e:
                print(f"警告：相机{idx+1}外参方向校验异常：{e}")

        if fixed_count > 0:
            print(f"完成外参方向自动修正：共修正 {fixed_count} 个相机外参")
