import cv2 as cv
import math
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from pathlib import Path
from values import (
    BOX_COLORS,
    BOX_THICKNESS,
    HAND_LANDMARKER_MODEL_FILE,
    HEART_DISPLAY_DURATION,
    HEART_FADE_DURATION,
    HEART_IMAGE_FILE,
    HEART_OVERLAY_SIZE,
    PERSON_LABEL,
    PERSON_LABEL_FONT_SCALE,
    PERSON_LABEL_PADDING,
    PERSON_LABEL_TEXT_COLOR,
    PERSON_LABEL_THICKNESS,
    YOLO_CONFIDENCE,
    YOLO_IMAGE_SIZE,
    YOLO_MIN_TRACK_HITS,
    YOLO_MODEL_FILE,
    TRACK_COLOR_REUSE_IOU,
    TRACK_STATE_MAX_AGE,
)

class PersonTracker:
    def __init__(self):
        hand = Path(__file__).with_name(HAND_LANDMARKER_MODEL_FILE)
        if not hand.is_file():
            raise FileNotFoundError(
                f'MediaPipe hand model not found: {hand}.'
            )
        self.model = YOLO(YOLO_MODEL_FILE)
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)
        self.colors = {}
        self.track_states = {}
        self.frame_number = 0
        self.heart_gesture_active = False
        self.heart_display_started = None
        self.heart_position = None

        image = Path(__file__).parent / 'images' / HEART_IMAGE_FILE
        self.heart_image = cv.imread(str(image), cv.IMREAD_COLOR)

        if self.heart_image is None:
            raise FileNotFoundError(f'Could not load heart image: {image}')

    @staticmethod
    def heart_gesture(hands):
        if len(hands) < 2:
            return False
        left, right = sorted(hands[:2], key=lambda hand: hand[0].x)

        hand_size = sum(
            math.hypot(hand[9].x - hand[0].x, hand[9].y - hand[0].y)
            for hand in (left, right)
        ) / 2

        if hand_size == 0:
            return False

        index_gap = math.hypot(left[8].x - right[8].x, left[8].y - right[8].y)
        thumb_gap = math.hypot(left[4].x - right[4].x, left[4].y - right[4].y)
        index_y = (left[8].y + right[8].y) / 2
        thumb_y = (left[4].y + right[4].y) / 2

        return (
            index_gap < hand_size * 0.5
            and thumb_gap < hand_size * 0.5
            and index_y < thumb_y - hand_size * 0.1
        )

    @staticmethod
    def box_iou(first, second):
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def detect(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            verbose=False,
        )
        annotated_frame = frame.copy()
        self.frame_number += 1

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = time.monotonic_ns() // 1_000_000
        hand_result = self.hand_detector.detect_for_video(mp_frame, timestamp_ms)

        gesture_detected = self.heart_gesture(hand_result.hand_landmarks)
        if gesture_detected:
            left, right = sorted(
                hand_result.hand_landmarks[:2],
                key=lambda hand: hand[0].x,
            )
            frame_height, frame_width = frame.shape[:2]
            self.heart_position = (
                int(
                    (left[4].x + right[4].x + left[8].x + right[8].x)
                    * frame_width
                    / 4
                ),
                int(
                    (left[4].y + right[4].y + left[8].y + right[8].y)
                    * frame_height
                    / 4
                ),
            )
            if not self.heart_gesture_active:
                self.heart_display_started = time.monotonic()
        self.heart_gesture_active = gesture_detected

        if self.heart_display_started is not None and self.heart_position is not None:
            elapsed = time.monotonic() - self.heart_display_started
            if elapsed >= HEART_DISPLAY_DURATION:
                self.heart_display_started = None
                self.heart_position = None
            else:
                overlay_size = min(
                    HEART_OVERLAY_SIZE,
                    frame.shape[1],
                    frame.shape[0],
                )
                heart = cv.resize(
                    self.heart_image,
                    (overlay_size, overlay_size),
                    interpolation=cv.INTER_AREA,
                )
                center_x, center_y = self.heart_position
                x = max(
                    0,
                    min(center_x - overlay_size // 2, frame.shape[1] - overlay_size),
                )
                y = max(
                    0,
                    min(center_y - overlay_size // 2, frame.shape[0] - overlay_size),
                )
                alpha = min(
                    1.0,
                    (HEART_DISPLAY_DURATION - elapsed) / HEART_FADE_DURATION,
                )
                alpha = max(0.0, alpha)
                region = annotated_frame[y:y + overlay_size, x:x + overlay_size]
                cv.addWeighted(heart, alpha, region, 1 - alpha, 0, region)

        detections = []
        fallback_indices = {}
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                fallback_index = fallback_indices.get(class_id, 0)
                fallback_indices[class_id] = fallback_index + 1
                track_id = (
                    int(box.id[0].item())
                    if box.id is not None
                    else fallback_index
                )
                detections.append(
                    ((class_id, track_id), class_id, (x1, y1, x2, y2))
                )

        current_keys = {detection[0] for detection in detections}
        self.track_states = {
            key: state
            for key, state in self.track_states.items()
            if self.frame_number - state[2] <= TRACK_STATE_MAX_AGE
        }
        matched_previous_keys = set()
        visible_detections = []

        for color_key, class_id, bounds in detections:
            state = self.track_states.get(color_key)
            if state is None:
                match = None
                match_iou = TRACK_COLOR_REUSE_IOU
                for previous_key, previous_state in self.track_states.items():
                    if (
                        previous_key in current_keys
                        or previous_key in matched_previous_keys
                        or previous_key[0] != class_id
                    ):
                        continue
                    overlap = self.box_iou(bounds, previous_state[0])
                    if overlap > match_iou:
                        match = (previous_key, previous_state)
                        match_iou = overlap

                if match is None:
                    color = self.colors.setdefault(
                        color_key,
                        BOX_COLORS[len(self.colors) % len(BOX_COLORS)],
                    )
                    hits = 1
                    confirmed = False
                else:
                    previous_key, state = match
                    matched_previous_keys.add(previous_key)
                    _, color, last_seen, previous_hits, was_confirmed = state
                    hits = previous_hits + 1 if last_seen == self.frame_number - 1 else 1
                    confirmed = was_confirmed and last_seen == self.frame_number - 1
                    self.colors[color_key] = color
                    del self.track_states[previous_key]
            else:
                _, color, last_seen, previous_hits, was_confirmed = state
                hits = previous_hits + 1 if last_seen == self.frame_number - 1 else 1
                confirmed = was_confirmed and last_seen == self.frame_number - 1

            confirmed = confirmed or hits >= YOLO_MIN_TRACK_HITS
            self.track_states[color_key] = (
                bounds,
                color,
                self.frame_number,
                hits,
                confirmed,
            )
            if confirmed:
                visible_detections.append((class_id, bounds, color))

        for class_id, (x1, y1, x2, y2), color in visible_detections:
            cv.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                BOX_THICKNESS,
            )

            if class_id == 0:
                label = PERSON_LABEL
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = PERSON_LABEL_FONT_SCALE
                thickness = PERSON_LABEL_THICKNESS
                padding = PERSON_LABEL_PADDING

                (text_width, text_height), baseline = cv.getTextSize(
                    label, font, font_scale, thickness
                )
                label_width = text_width + padding * 2
                label_height = text_height + baseline + padding * 2
                frame_width = annotated_frame.shape[1]
                label_x = max(
                    0,
                    min(x1, frame_width - label_width),
                )
                label_y = max(0, y1 - label_height)

                cv.rectangle(
                    annotated_frame,
                    (label_x, label_y),
                    (label_x + label_width, label_y + label_height),
                    color, -1,
                )
                cv.putText(
                    annotated_frame,
                    label,
                    (label_x + padding, label_y + padding + text_height),
                    font,
                    font_scale,
                    PERSON_LABEL_TEXT_COLOR,
                    thickness,
                    cv.LINE_AA,
                )
        return annotated_frame