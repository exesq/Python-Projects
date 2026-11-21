import threading
import tkinter as tk

import cv2 as cv
from motion_detection import PersonTracker
from PIL import Image, ImageTk
from values import EXTRA_HEIGHT, WINDOW_WIDTH


class FrameCapture(threading.Thread):
    def __init__(self, camera):
        super().__init__(daemon=True)
        self.camera = camera
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.error = None
        self.stopped = threading.Event()

    def run(self):
        try:
            while not self.stopped.is_set():
                ok, current = self.camera.read()
                if not ok:
                    if self.stopped.is_set():
                        break
                    raise RuntimeError('Could not read from the camera.')

                with self.lock:
                    self.latest_frame = current
                    self.latest_frame_id += 1
        except Exception as error:
            with self.lock:
                self.error = error
        finally:
            self.camera.release()

    def latest(self):
        with self.lock:
            return self.latest_frame, self.latest_frame_id, self.error

    def stop(self):
        self.stopped.set()

class FrameProcessor(threading.Thread):
    def __init__(self, camera, tracker):
        super().__init__(daemon=True)
        self.capture = FrameCapture(camera)
        self.tracker = tracker
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_id = 0
        self.error = None
        self.stopped = threading.Event()

    def run(self):
        self.capture.start()
        last_captured_frame_id = 0
        try:
            while not self.stopped.is_set():
                current, frame_id, error = self.capture.latest()
                if error is not None:
                    raise RuntimeError('Could not capture the camera frame.') from error
                if current is None or frame_id == last_captured_frame_id:
                    self.stopped.wait(0.005)
                    continue

                current = cv.resize(current, (640, 360), interpolation=cv.INTER_AREA)
                current = cv.flip(current, 1)
                frame = self.tracker.detect(current)
                with self.lock:
                    self.latest_frame = frame
                    self.latest_frame_id += 1
                last_captured_frame_id = frame_id
        except Exception as error:
            with self.lock:
                self.error = error
        finally:
            self.capture.stop()
            self.capture.join()

    def latest(self):
        with self.lock:
            return self.latest_frame, self.latest_frame_id, self.error

    def stop(self):
        self.stopped.set()

def run():
    tracker = PersonTracker()
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError('Could not open the camera.')

    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

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
        last_displayed_frame_id = 0

        def show_frame():
            nonlocal last_displayed_frame_id
            frame, frame_id, error = processor.latest()
            if error is not None:
                root.destroy()
                raise RuntimeError('Could not process the camera frame.') from error
            if frame is None or frame_id == last_displayed_frame_id:
                root.after(15, show_frame)
                return

            target_width, target_height = preview.winfo_width(), preview.winfo_height()
            if target_width <= 1 or target_height <= 1:
                root.after(50, show_frame)
                return

            height, width = frame.shape[:2]
            scale = max(target_width / width, target_height / height)
            width, height = int(width * scale), int(height * scale)
            interpolation = cv.INTER_LINEAR if scale > 1 else cv.INTER_AREA
            frame = cv.resize(frame, (width, height), interpolation=interpolation)
            left = (width - target_width) // 2
            top = (height - target_height) // 2
            frame = frame[top:top + target_height, left:left + target_width]
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(frame))
            preview.configure(image=photo)
            preview.image = photo
            last_displayed_frame_id = frame_id
            root.after(15, show_frame)

        def toggle_fullscreen(event=None):
            root.attributes('-fullscreen', not root.attributes('-fullscreen'))
            return 'break'

        def exit_fullscreen(event=None):
            root.attributes('-fullscreen', False)
            return 'break'

        def close(event=None):
            processor.stop()
            root.destroy()

        root.protocol('WM_DELETE_WINDOW', close)
        root.bind_all('<KeyPress-q>', close)
        root.bind('<F11>', toggle_fullscreen)
        root.bind('<Escape>', exit_fullscreen)
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