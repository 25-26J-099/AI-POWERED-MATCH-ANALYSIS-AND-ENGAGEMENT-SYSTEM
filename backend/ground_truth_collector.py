# ground_truth_collector.py
import cv2
import json
from collections import defaultdict

class GTCollector:
    """Interactive tool for collecting ground truth player IDs."""
    
    def __init__(self, video_path, output_path='ground_truth.json'):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(video_path)
        self.ground_truth = defaultdict(dict)  # {frame_idx: {track_id: gt_id}}
        self.current_frame = 0
        self.current_detections = []
        
    def annotate_frame(self, frame_idx, detections):
        """
        Annotate frame with ground truth IDs.
        detections: list of (track_id, bbox)
        """
        self.current_frame = frame_idx
        self.current_detections = detections
        
        # Read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Draw bboxes with track IDs
        for track_id, bbox in detections:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Track: {track_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Annotate Frame', frame)
        
        print(f"\nFrame {frame_idx}")
        print("Enter ground truth IDs (track_id:gt_id), or 'skip', 'save', 'quit':")
        
        while True:
            cmd = input("> ").strip()
            
            if cmd == 'skip':
                break
            elif cmd == 'save':
                self.save()
            elif cmd == 'quit':
                self.save()
                return False
            else:
                # Parse track_id:gt_id
                try:
                    track_id, gt_id = map(int, cmd.split(':'))
                    self.ground_truth[frame_idx][track_id] = gt_id
                    print(f"Recorded: Track {track_id} = GT {gt_id}")
                except:
                    print("Invalid format. Use: track_id:gt_id")
        
        return True
    
    def save(self):
        with open(self.output_path, 'w') as f:
            # Convert defaultdict to regular dict for JSON
            gt_dict = {str(k): dict(v) for k, v in self.ground_truth.items()}
            json.dump(gt_dict, f, indent=2)
        print(f"Saved to {self.output_path}")

# Usage:
# collector = GTCollector('input_video.mp4')
# # From your pipeline, pass detections
# collector.annotate_frame(frame_idx=100, detections=[(1, (x1,y1,x2,y2)), ...])