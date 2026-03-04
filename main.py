import argparse
import cv2
import time
from core.camera import CameraStream
from core.detector import Detector
from core.tracker import Tracker
from core.director import Director
from core.humor import HumorEngine
from core.renderer import Renderer

def parse_args():
    parser = argparse.ArgumentParser(description="GOSSIP: Observation System for Pedestrians")
    parser.add_argument(
        '--source', 
        type=str, 
        default='0',
        help="Source of video stream. Use '0' for webcam, or path to video file (e.g., 'assets/video.mp4')"
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help="Path to save the output video (e.g., 'assets/demo.mp4')"
    )
    return parser.parse_args()
def main():
    args = parse_args()
    print("Initializing GOSSIP: General Observation System for Spontaneous Interpretation of Pedestrians")
    
    # Process source argument (convert string '0' to int 0 for webcam)
    video_src = int(args.source) if args.source.isdigit() else args.source
    
    try:
        camera = CameraStream(src=video_src).start()
    except Exception as e:
        print(f"Error connecting to camera feed: {e}")
        return
    # By default, downloads the nano weights to standard Ultralytics directory
    detector = Detector(model_path='yolov8n.pt')
    
    # Instantiate the logic filtering sequence
    tracker = Tracker()
    director = Director(max_actors=3, history_thresh=10, cooldown_sec=2.0)
    humor = HumorEngine(json_path='data/quotes.json', update_interval=4.0)
    renderer = Renderer(alpha=0.9)
    
    # Wait for the camera buffer to warm up frames
    if isinstance(video_src, int):
        time.sleep(1.0)
    
    print(f"Starting Main Event Loop with source '{args.source}'... (Press 'q' inside output window to terminate)")
    
    video_writer = None
    if args.save:
        ret, frame = camera.read()
        if ret:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = getattr(camera, 'fps', 30.0)
            video_writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
            print(f"Saving output to {args.save}")
    
    # Enable Fullscreen Window Properties
    cv2.namedWindow("GOSSIP Feedback Monitor", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("GOSSIP Feedback Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    frame_count = 0
    inference_interval = 3  # Target ~10 FPS YOLO inferencing to save CPU cycles
    last_detections = []

    while True:
        # 1. Fetch
        ret, frame = camera.read()
        if not ret:
            print("Video stream ended or disconnected.")
            break
            
        # Optional: Aggressive downscale for testing purposes
        # frame = cv2.resize(frame, (1280, 720))
            
        # 2. Detect (Interleaved async approximation)
        if frame_count % inference_interval == 0:
            last_detections = detector.detect(frame, conf=0.35)
            
        # 3. Track (Linking bounding boxes over time)
        tracks = tracker.update(last_detections)
        
        # 4. Direct (Filter noise and select priority Actors)
        actors = director.rank_and_select(tracks)
        
        # 5. Script (Attach deterministic humor quotes to chosen tags)
        quotes = humor.update(actors)
        
        # Render
        out_frame = renderer.render(frame, tracks, actors, quotes)
        
        # Display the result
        cv2.imshow("GOSSIP Feedback Monitor", out_frame)
        
        if video_writer:
            video_writer.write(out_frame)
            
        # Event bindings
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Terminating application...")
            break
            
        frame_count += 1
        
    camera.stop()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
