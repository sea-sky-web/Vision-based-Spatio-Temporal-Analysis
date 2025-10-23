import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project.data.wildtrack_loader import WildtrackDataset

CFG = {
    'DATA': {
        'DATA_ROOT': r'c:\Users\zhangweichao\Desktop\BEV_fursion\Vision-based-Spatio-Temporal-Analysis\project\data\Wildtrack',
        'VIEWS': 7,
        'IMG_SIZE': [3, 480, 640],
    }
}

if __name__ == '__main__':
    ds = WildtrackDataset(CFG)
    print('Dataset OK:', len(ds), 'frames')
    print('Views:', ds.views)
    print('First calib shapes:', ds.intrinsics[0][0].shape, ds.extrinsics[0][0].shape)
    print('First frame file:', ds.frame_files[0] if len(ds.frame_files) > 0 else 'N/A')