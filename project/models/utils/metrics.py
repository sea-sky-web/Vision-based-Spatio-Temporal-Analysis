import motmetrics as mm
import numpy as np
from typing import List, Dict, Any

def calculate_iou(boxA, boxB):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class MOTMetrics:
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, gt_tracks: List[Dict[str, Any]], ts_tracks: List[Dict[str, Any]]):
        """
        Updates the accumulator with data from a single frame.
        :param gt_tracks: A list of ground-truth tracks for the current frame.
        :param ts_tracks: A list of tracked objects from the tracker.
        """
        gt_ids = [t['track_id'] for t in gt_tracks]
        ts_ids = [t['track_id'] for t in ts_tracks]

        gt_boxes = [t['box'] for t in gt_tracks]
        ts_boxes = [t['box'] for t in ts_tracks]

        # Calculate the distance matrix (1 - IoU)
        distance_matrix = np.zeros((len(gt_boxes), len(ts_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, ts_box in enumerate(ts_boxes):
                distance_matrix[i, j] = 1 - calculate_iou(gt_box, ts_box)

        self.acc.update(gt_ids, ts_ids, distance_matrix)

    def summary(self) -> Dict[str, Any]:
        """
        Computes and returns the tracking metrics.
        :return: A dictionary containing the MOT metrics.
        """
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=mm.metrics.motchallenge_metrics, name='acc')

        # Convert to a more friendly format
        summary_dict = summary.to_dict('index')['acc']
        return summary_dict
