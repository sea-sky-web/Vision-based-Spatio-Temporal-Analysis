from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import xml.etree.ElementTree as ET

import torch
from PIL import Image
import torchvision.transforms as T

from .transforms import build_transforms


def _parse_float_list(text: str) -> List[float]:
    """Parse a list of floats from a free-form text string (comma/space/semi/line separated)."""
    if text is None:
        return []
    # Replace common separators with spaces
    cleaned = re.sub(r"[\,;\n\t]+", " ", text)
    parts = [p for p in cleaned.strip().split(" ") if p != ""]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            # Ignore non-numeric tokens
            continue
    return vals


def _reshape(vals: List[float], rows: int, cols: int) -> torch.Tensor:
    if len(vals) < rows * cols:
        raise ValueError(f"Not enough values to reshape: need {rows*cols}, got {len(vals)}")
    return torch.tensor(vals[: rows * cols], dtype=torch.float32).reshape(rows, cols)


def _try_get_matrix(root: ET.Element, tag_names: List[str], shape: Tuple[int, int]) -> Optional[torch.Tensor]:
    """Try to find a matrix under any of the tag names. Supports nested <data> nodes or raw text."""
    rows, cols = shape
    for name in tag_names:
        for elem in root.findall(f".//{name}"):
            # Opencv XML may store <name><rows/><cols/><data>...</data></name>
            data_elem = elem.find("data")
            if data_elem is not None and (data_elem.text is not None):
                vals = _parse_float_list(data_elem.text)
                if len(vals) >= rows * cols:
                    return _reshape(vals, rows, cols)
            # Raw text inside tag
            if elem.text is not None:
                vals = _parse_float_list(elem.text)
                if len(vals) >= rows * cols:
                    return _reshape(vals, rows, cols)
            # OpenCV style with nested elements <_><_>
            text_all = " ".join([e.text or "" for e in elem.iter()])
            vals = _parse_float_list(text_all)
            if len(vals) >= rows * cols:
                return _reshape(vals, rows, cols)
    return None


def _load_camera_xml(xml_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load K (3x3) and Rt (4x4) from a camera XML file with flexible tag names."""
    root = ET.parse(str(xml_path)).getroot()
    # Try common tag name variants
    K_tags = [
        "K", "intrinsic", "intrinsics", "camera_matrix", "IntrinsicMatrix", "MatrixK", "A"
    ]
    R_tags = [
        "R", "rotation", "RotationMatrix", "rotation_matrix"
    ]
    T_tags = [
        "T", "translation", "TranslationVector", "t"
    ]
    RT_tags = [
        "RT", "ExtrinsicMatrix", "Pose", "MatrixRT"
    ]

    K = _try_get_matrix(root, K_tags, (3, 3))
    Rt = _try_get_matrix(root, RT_tags, (3, 4))

    if Rt is None:
        R = _try_get_matrix(root, R_tags, (3, 3))
        t = _try_get_matrix(root, T_tags, (3, 1))
        if R is not None and t is not None:
            Rt34 = torch.cat([R, t], dim=1)  # [3,4]
        else:
            Rt34 = None
    else:
        Rt34 = Rt

    if K is None:
        # Provide a safe default
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = K[1, 1] = 1000.0

    if Rt34 is None:
        # Fall back to identity extrinsic
        Rt44 = torch.eye(4, dtype=torch.float32)
    else:
        Rt44 = torch.eye(4, dtype=torch.float32)
        Rt44[:3, :4] = Rt34

    return K, Rt44


def _discover_camera_xmls(calib_dir: Path, views: int) -> List[Optional[Path]]:
    """Return list of XML paths per camera index (1..views). Match by filename containing C{i} or {i}."""
    if not calib_dir.exists():
        return [None] * views
    xmls = list(calib_dir.rglob("*.xml"))
    per_view: List[Optional[Path]] = [None] * views
    for i in range(1, views + 1):
        # Prefer exact 'C{i}.xml' or '*C{i}*.xml'
        candidates = [p for p in xmls if re.search(fr"(^|[^\w])C{i}([^\w]|$)", p.stem, flags=re.IGNORECASE)]
        if not candidates:
            candidates = [p for p in xmls if re.search(fr"(^|[^\w]){i}([^\w]|$)", p.stem)]
        per_view[i - 1] = candidates[0] if candidates else None
    return per_view


def _load_wildtrack_calibrations(calib_root: Path, views: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # 选择内参/外参目录
    intr_dir = calib_root / 'intrinsic_original' if (calib_root / 'intrinsic_original').exists() else (
        calib_root / 'intrinsic_zero' if (calib_root / 'intrinsic_zero').exists() else calib_root
    )
    extr_dir = calib_root / 'extrinsic' if (calib_root / 'extrinsic').exists() else calib_root

    # Wildtrack默认7机位顺序
    default_names = ['CVLab1', 'CVLab2', 'CVLab3', 'CVLab4', 'IDIAP1', 'IDIAP2', 'IDIAP3']
    if views == 7:
        cam_names = default_names
    else:
        # 非7视角时，按目录里可用的命名排序（CVLab*优先，其次IDIAP*，最后其他）
        candidates = [p.stem for p in list(intr_dir.rglob('*.xml')) + list(extr_dir.rglob('*.xml'))]
        names = set()
        for s in candidates:
            m = re.search(r'(CVLab\d+|IDIAP\d+)', s, flags=re.IGNORECASE)
            if m: names.add(m.group(1))
        cam_names = sorted([n for n in names if n.lower().startswith('cvlab')]) + \
                    sorted([n for n in names if n.lower().startswith('idiap')])
        if len(cam_names) < views:
            # 补足/截断
            nums = [str(i) for i in range(1, views + 1)]
            cam_names += [f'Cam{i}' for i in nums[len(cam_names):]]
            cam_names = cam_names[:views]

    Ks: List[torch.Tensor] = []
    Rts: List[torch.Tensor] = []

    for name in cam_names:
        # 找内参文件
        intr_xmls = list(intr_dir.rglob('*.xml'))
        intr_match = next((p for p in intr_xmls if re.search(fr'{name}', p.stem, flags=re.IGNORECASE)), None)
        # 找外参文件
        extr_xmls = list(extr_dir.rglob('*.xml'))
        extr_match = next((p for p in extr_xmls if re.search(fr'{name}', p.stem, flags=re.IGNORECASE)), None)

        if intr_match is None:
            print(f"[WildtrackDataset] 警告: 未找到相机 {name} 的内参XML，使用默认K。")
            K = torch.eye(3, dtype=torch.float32); K[0,0] = K[1,1] = 1000.0
        else:
            K_only = _try_get_matrix(ET.parse(str(intr_match)).getroot(), ['K','intrinsic','camera_matrix','IntrinsicMatrix','MatrixK','A'], (3,3))
            if K_only is None:
                print(f"[WildtrackDataset] 警告: 相机 {name} 的内参XML未解析到K，使用默认K: {intr_match}")
                K = torch.eye(3, dtype=torch.float32); K[0,0] = K[1,1] = 1000.0
            else:
                K = K_only

        if extr_match is None:
            print(f"[WildtrackDataset] 警告: 未找到相机 {name} 的外参XML，使用单位Rt。")
            Rt = torch.eye(4, dtype=torch.float32)
        else:
            root_ex = ET.parse(str(extr_match)).getroot()
            Rt34 = _try_get_matrix(root_ex, ['RT','ExtrinsicMatrix','Pose','MatrixRT'], (3,4))
            if Rt34 is None:
                R = _try_get_matrix(root_ex, ['R','rotation','RotationMatrix','rotation_matrix'], (3,3))
                t = _try_get_matrix(root_ex, ['T','translation','TranslationVector','t'], (3,1))
                if R is not None and t is not None:
                    Rt34 = torch.cat([R, t], dim=1)
            if Rt34 is None:
                print(f"[WildtrackDataset] 警告: 相机 {name} 的外参XML未解析到Rt，使用单位Rt: {extr_match}")
                Rt = torch.eye(4, dtype=torch.float32)
            else:
                Rt = torch.eye(4, dtype=torch.float32)
                Rt[:3, :4] = Rt34

        Ks.append(K)
        Rts.append(Rt)

    return Ks, Rts


class WildtrackDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.data_root = Path(cfg['DATA']['DATA_ROOT']).resolve()
        self.views = int(cfg['DATA']['VIEWS'])
        C, H, W = cfg['DATA']['IMG_SIZE']
        self.img_size = (H, W)
        self.transform = build_transforms(img_size=(H, W))
        self.frame_files: List[str] = []
        self.cam_dirs: List[Path] = []
        self.intrinsics: List[List[torch.Tensor]] = []
        self.extrinsics: List[List[torch.Tensor]] = []
        self.annotations_dir = self.data_root / 'Annotations'

        # Discover camera image directories
        img_root = self.data_root / 'Image_subsets'
        if not img_root.exists():
            raise FileNotFoundError(f"未找到图像根目录: {img_root}")
        for i in range(1, self.views + 1):
            d = img_root / f'C{i}'
            if not d.exists():
                raise FileNotFoundError(f"未找到相机文件夹: {d}")
            self.cam_dirs.append(d)
        # Frame files follow the first camera's files
        self.frame_files = sorted([p.name for p in self.cam_dirs[0].iterdir() if p.is_file()])
        if len(self.frame_files) == 0:
            raise FileNotFoundError("未发现图像文件")

        # Parse camera calibrations (XML)
        calib_dir_candidates = [self.data_root / 'Calibration', self.data_root / 'Calibrations', self.data_root / 'calibration']
        calib_dir = next((d for d in calib_dir_candidates if d.exists()), None)
        if calib_dir is None:
            raise FileNotFoundError("未找到标定目录(尝试: Calibration/Calibrations/calibration)。请确保Wildtrack标定XML在数据根目录下。")

        # 替换为按目录分别读取内外参，并按相机顺序组合
        Ks, Rts = _load_wildtrack_calibrations(calib_dir, self.views)
        # 使用批量解析的结果，不再逐相机查找单文件XML

        # Replicate calibrations per frame (static per camera)
        self.intrinsics = [Ks for _ in range(len(self.frame_files))]
        self.extrinsics = [Rts for _ in range(len(self.frame_files))]

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        imgs = []
        paths = []
        for v in range(self.views):
            p = self.cam_dirs[v] / self.frame_files[idx]
            img = Image.open(p).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
            paths.append(str(p))
        images = torch.stack(imgs, dim=0)  # [V,3,H,W]
        calib = {
            'intrinsic': self.intrinsics[idx],
            'extrinsic': self.extrinsics[idx],
        }
        targets = {
            'boxes_world': torch.zeros(0, 4),
            'centers_world': torch.zeros(0, 2),
            'keypoints': None,
            'calib': {'intrinsic': self.intrinsics[idx], 'extrinsic': self.extrinsics[idx]},
        }
        meta = {'frame_idx': int(idx), 'paths': paths}
        return {'images': images, 'calib': calib, 'targets': targets, 'meta': meta}


def collate_fn(batch: List[Dict[str, Any]]):
    batch = [b for b in batch if b is not None]
    images = torch.stack([b['images'] for b in batch], dim=0)  # [B,V,3,H,W]
    calib_intrinsic = [b['calib']['intrinsic'] for b in batch]
    calib_extrinsic = [b['calib']['extrinsic'] for b in batch]
    targets = [b['targets'] for b in batch]
    meta = [b['meta'] for b in batch]
    return {
        'images': images,
        'calib': {'intrinsic': calib_intrinsic, 'extrinsic': calib_extrinsic},
        'targets': targets,
        'meta': meta,
    }