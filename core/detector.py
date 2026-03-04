from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize the YOLO model for inference."""
        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.35):
        """
        Run inference and return detected bounding boxes for people.
        Returns: list of [x1, y1, x2, y2, confidence]
        """
        # Set classes=[0] to filter for person class only
        results = self.model(frame, classes=[0], conf=conf, verbose=False)
        boxes = []
        
        for result in results:
            for box in result.boxes:
                # box.xyxy provides the [x1, y1, x2, y2] coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].cpu().numpy()
                boxes.append([int(x1), int(y1), int(x2), int(y2), float(score)])
                
        return boxes
