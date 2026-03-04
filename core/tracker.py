def bb_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

class Tracker:
    def __init__(self, max_lost=5, iou_thresh=0.3):
        """
        Simple IoU tracker for fast, CPU-friendly ID persistence.
        """
        self.next_id = 0
        self.tracks = {}  # tid: {'box': [x1,y1,x2,y2], 'lost': int, 'history': int}
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh

    def update(self, detections):
        """
        Update the tracker with new detections.
        detections: list of [x1, y1, x2, y2, conf]
        """
        updated_tracks = {}
        unmatched_dets = list(range(len(detections)))
        
        # Greedy Match
        for track_id, track_data in self.tracks.items():
            best_iou = self.iou_thresh
            best_match_idx = -1
            
            for i in unmatched_dets:
                iou = bb_iou(track_data['box'], detections[i][:4])
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i
                    
            if best_match_idx != -1:
                # Successfully tracked
                updated_tracks[track_id] = {
                    'box': detections[best_match_idx][:4],
                    'lost': 0,
                    'history': track_data['history'] + 1
                }
                unmatched_dets.remove(best_match_idx)
            else:
                # Track lost but within grace period
                track_data['lost'] += 1
                if track_data['lost'] <= self.max_lost:
                    updated_tracks[track_id] = track_data

        # Add remaining unmatched detections as new tracks
        for i in unmatched_dets:
            updated_tracks[self.next_id] = {
                'box': detections[i][:4],
                'lost': 0,
                'history': 1
            }
            self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks
