import base64
import threading
import tkinter as tk
import cv2 as cv
from motion_detection import PersonTracker
from values import EXTRA_HEIGHT, WINDOW_WIDTH

class FrameProcessor(threading.Thread):
    def __init__(self, camera, tracker):
        super().__init__(daemon=True)
        self.camera = camera
        self.tracker = tracker
        self.lock = threading.Lock()
        self.latest_frame = None
        self.error = None
        self.stopped = threading.Event()

    def run(self):
        try:
            while not self.stopped.is_set():
                ok, current = self.camera.read()
                if not ok:
                    raise RuntimeError('Could not read from the camera.')

                current = cv.flip(current, 1)
                frame = self.tracker.detect(current)
                with self.lock:
                    self.latest_frame = frame
        except Exception as error:
            with self.lock:
                self.error = error
        finally:
            self.camera.release()

    def latest(self):
        with self.lock:
            return self.latest_frame, self.error

    def stop(self):
        self.stopped.set()

def run():
    tracker = PersonTracker()
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError('Could not open the camera.')

    processor = None
    try:
        ok, initial_frame = camera.read()
        if not ok:
            raise RuntimeError('Could not read from the camera.')

        root = tk.Tk()
        root.title('Motion Tracking Camera')
        screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
        frame_height, frame_width = initial_frame.shape[:2]
        window_height = int(WINDOW_WIDTH * frame_height / frame_width) + EXTRA_HEIGHT
        root.geometry(
            f'{WINDOW_WIDTH}x{window_height}+'
            f'{(screen_width - WINDOW_WIDTH) // 2}+'
            f'{(screen_height - window_height) // 2}'
        )
        preview = tk.Label(root, bg='black')
        preview.pack(fill=tk.BOTH, expand=True)
        processor = FrameProcessor(camera, tracker)
        processor.start()

        def show_frame():
            frame, error = processor.latest()
            if error is not None:
                root.destroy()
                raise RuntimeError('Could not process the camera frame.') from error
            if frame is None:
                root.after(15, show_frame)
                return
            target_width, target_height = preview.winfo_width(), preview.winfo_height()
            if target_width <= 1 or target_height <= 1:
                root.after(50, show_frame)
                return

            height, width = frame.shape[:2]
            scale = max(target_width / width, target_height / height)
            width, height = int(width * scale), int(height * scale)
            frame = cv.resize(frame, (width, height))
            left, top = (width - target_width) // 2, (height - target_height) // 2
            frame = frame[top : top + target_height, left : left + target_width]

            ok, encoded = cv.imencode('.png', frame)
            if not ok:
                raise RuntimeError('Could not encode the camera frame.')
            image = tk.PhotoImage(
                data=base64.b64encode(encoded).decode('ascii'), format='png'
            )
            preview.configure(image=image)
            preview.image = image
            root.after(15, show_frame)

        def close(event=None):
            processor.stop()
            root.destroy()

        root.protocol('WM_DELETE_WINDOW', close)
        root.bind_all('<KeyPress-q>', close)
        root.after(100, show_frame)
        root.mainloop()
    finally:
        if processor is not None:
            processor.stop()
            processor.join()
        else:
            camera.release()

if __name__ == '__main__':
    run()