import cv2
import threading
import time

class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        
        self.is_file = isinstance(src, str)
        if self.is_file:
            self.fps = self.stream.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0
            self.frame_delay = 1.0 / self.fps

    def start(self):
        """Start the thread to read frames from the video stream."""
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        """Keep looping indefinitely until the thread is stopped."""
        while not self.stopped:
            start_time = time.time()
            
            ret, frame = self.stream.read()
            if not ret:
                self.stop()
                break
            with self.lock:
                self.ret = ret
                self.frame = frame
                
            if self.is_file:
                elapsed = time.time() - start_time
                time_to_sleep = self.frame_delay - elapsed
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

    def read(self):
        """Return the frame most recently read."""
        with self.lock:
            if not self.ret:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        """Indicate that the thread should be stopped."""
        self.stopped = True
        self.stream.release()
