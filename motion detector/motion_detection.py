import math
import time
from pathlib import Path

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from values import (
    BOX_COLORS,
    BOX_THICKNESS,
    FACE_DETECTOR_MODEL_FILE,
    FEMALE_LABEL,
    GENDER_CLASSIFIER_MODEL_FILE,
    GENDER_MIN_CONFIDENCE,
    GENDER_RECHECK_INTERVAL,
    GESTURE_CONFIRMATION_FRAMES,
    GESTURE_OVERLAY_SIZE,
    HAND_LANDMARKER_MODEL_FILE,
    HEART_DISPLAY_DURATION,
    HEART_FADE_DURATION,
    HEART_IMAGE_FILE,
    MALE_LABEL,
    MAX_GESTURE_PEOPLE,
    PEACE_IMAGE_FILE,
    PERSON_LABEL,
    PERSON_LABEL_FONT_SCALE,
    PERSON_LABEL_PADDING,
    PERSON_LABEL_TEXT_COLOR,
    PERSON_LABEL_THICKNESS,
    ROCK_AND_ROLL_IMAGE_FILE,
    SHAKA_IMAGE_FILE,
    TRACK_COLOR_REUSE_IOU,
    TRACK_STATE_MAX_AGE,
    YOLO_CONFIDENCE,
    YOLO_IMAGE_SIZE,
    YOLO_MIN_TRACK_HITS,
    YOLO_MODEL_FILE,
)


class PersonTracker:
    def __init__(self):
        hand = Path(__file__).with_name(HAND_LANDMARKER_MODEL_FILE)
        if not hand.is_file():
            raise FileNotFoundError(
                f'MediaPipe hand model not found: {hand}.'
            )
        self.model = YOLO(str(Path(__file__).with_name(YOLO_MODEL_FILE)))
        gender_model = Path(__file__).with_name(GENDER_CLASSIFIER_MODEL_FILE)
        if not gender_model.is_file():
            raise FileNotFoundError(
                f'Gender classifier model not found: {gender_model}'
            )
        self.gender_model = YOLO(str(gender_model))
        face_model = Path(__file__).with_name(FACE_DETECTOR_MODEL_FILE)
        if not face_model.is_file():
            raise FileNotFoundError(
                f'Face detector model not found: {face_model}'
            )
        self.face_detector = cv.CascadeClassifier(str(face_model))
        if self.face_detector.empty():
            raise RuntimeError(f'Could not load face detector model: {face_model}')
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_GESTURE_PEOPLE * 2,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)
        self.colors = {}
        self.track_states = {}
        self.frame_number = 0
        self.gesture_states = {}
        self.gesture_confirmation_frames = {}
        self.gender_labels = {}

        images = Path(__file__).parent / 'images'
        heart_image = images / HEART_IMAGE_FILE
        self.heart_image = cv.imread(str(heart_image), cv.IMREAD_COLOR)

        if self.heart_image is None:
            raise FileNotFoundError(f'Could not load heart image: {heart_image}')

        peace_image = images / PEACE_IMAGE_FILE
        self.peace_image = cv.imread(str(peace_image), cv.IMREAD_COLOR)

        if self.peace_image is None:
            raise FileNotFoundError(f'Could not load peace image: {peace_image}')

        rock_and_roll_image = images / ROCK_AND_ROLL_IMAGE_FILE
        self.rock_and_roll_image = cv.imread(str(rock_and_roll_image), cv.IMREAD_COLOR)
        if self.rock_and_roll_image is None:
            raise FileNotFoundError(
                f'Could not load rock and roll image: {rock_and_roll_image}'
            )

        surf_boy_image = images / SHAKA_IMAGE_FILE
        self.surf_boy_image = cv.imread(str(surf_boy_image), cv.IMREAD_COLOR)
        if self.surf_boy_image is None:
            raise FileNotFoundError(f'Could not load surf boy image: {surf_boy_image}')

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
    def peace_gesture(hand):
        if len(hand) < 21:
            return False

        return (
            hand[12].y < hand[10].y
            and hand[8].y < hand[6].y
            and hand[16].y > hand[14].y
            and hand[20].y > hand[18].y
        )

    @staticmethod
    def rock_and_roll_gesture(hand):
        if len(hand) < 21:
            return False

        return (
            hand[8].y < hand[6].y
            and hand[12].y > hand[10].y
            and hand[16].y > hand[14].y
            and hand[20].y < hand[18].y
        )

    @staticmethod
    def shaka_gesture(hand):
        if len(hand) < 21:
            return False

        hand_size = math.hypot(hand[9].x - hand[0].x, hand[9].y - hand[0].y)
        thumb_spread = math.hypot(hand[4].x - hand[5].x, hand[4].y - hand[5].y)
        return (
            hand_size > 0
            and thumb_spread > hand_size * 0.5
            and hand[8].y > hand[6].y
            and hand[12].y > hand[10].y
            and hand[16].y > hand[14].y
            and hand[20].y < hand[18].y
        )

    @staticmethod
    def _gesture_confirmation_count(detected, previous_count):
        return (
            min(previous_count + 1, GESTURE_CONFIRMATION_FRAMES)
            if detected
            else 0
        )

    @staticmethod
    def _person_for_hand(hand, people, frame_width, frame_height):
        palm_x = (hand[0].x + hand[9].x) * frame_width / 2
        palm_y = (hand[0].y + hand[9].y) * frame_height / 2
        matches = []
        for _, person_key, (x1, y1, x2, y2) in people:
            person_height = y2 - y1
            if person_height <= 0:
                continue

            distance_x = max(x1 - palm_x, 0, palm_x - x2)
            distance_y = max(y1 - palm_y, 0, palm_y - y2)
            distance = math.hypot(distance_x, distance_y)
            if distance <= person_height * 0.5:
                matches.append(
                    (
                        distance / person_height,
                        (x2 - x1) * person_height,
                        person_key,
                    )
                )
        return min(matches)[2] if matches else None

    @staticmethod
    def _draw_gesture_overlay(frame, image, position, display_started):
        if display_started is None or position is None:
            return display_started

        elapsed = time.monotonic() - display_started
        if elapsed >= HEART_DISPLAY_DURATION:
            return None

        overlay_size = min(GESTURE_OVERLAY_SIZE, frame.shape[1], frame.shape[0])
        overlay = cv.resize(
            image, (overlay_size, overlay_size), interpolation=cv.INTER_AREA
        )
        center_x, center_y = position
        x = max(0, min(center_x - overlay_size // 2, frame.shape[1] - overlay_size))
        y = max(0, min(center_y - overlay_size // 2, frame.shape[0] - overlay_size))
        alpha = min(1.0, (HEART_DISPLAY_DURATION - elapsed) / HEART_FADE_DURATION)
        region = frame[y:y + overlay_size, x:x + overlay_size]
        cv.addWeighted(overlay, alpha, region, 1 - alpha, 0, region)
        return display_started

    def _classify_gender(self, frame, bounds):
        x1, y1, x2, y2 = bounds
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        gray_crop = cv.cvtColor(person_crop, cv.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray_crop,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return None

        face_x, face_y, face_width, face_height = max(
            faces, key=lambda face: face[2] * face[3]
        )
        face_crop = person_crop[
            face_y:face_y + face_height,
            face_x:face_x + face_width,
        ]
        prediction = self.gender_model.predict(
            face_crop, imgsz=224, verbose=False
        )[0]
        if prediction.probs.top1conf.item() < GENDER_MIN_CONFIDENCE:
            return None

        gender = prediction.names[prediction.probs.top1].lower()
        return {'male': MALE_LABEL, 'female': FEMALE_LABEL}.get(gender)

    def _update_gender_labels(self, frame, people):
        current_person_keys = {person[1] for person in people}
        self.gender_labels = {
            key: state
            for key, state in self.gender_labels.items()
            if self.frame_number - state[2] <= TRACK_STATE_MAX_AGE
        }

        for _, person_key, bounds in people:
            cached = self.gender_labels.get(person_key)
            if cached is None:
                match = max(
                    (
                        (self.box_iou(bounds, state[3]), key, state)
                        for key, state in self.gender_labels.items()
                        if key not in current_person_keys
                        and key[0] == person_key[0]
                    ),
                    key=lambda candidate: candidate[0],
                    default=None,
                )
                if match is not None and match[0] > TRACK_COLOR_REUSE_IOU:
                    _, previous_key, cached = match
                    del self.gender_labels[previous_key]

            if cached is None:
                label = None
                last_checked = self.frame_number - GENDER_RECHECK_INTERVAL
            else:
                label, last_checked = cached[:2]

            if (
                label is None
                and self.frame_number - last_checked >= GENDER_RECHECK_INTERVAL
            ):
                label = self._classify_gender(frame, bounds)
                last_checked = self.frame_number

            self.gender_labels[person_key] = (
                label,
                last_checked,
                self.frame_number,
                bounds,
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
        detections = []
        fallback_indices = {}
        person_candidates = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                fallback_index = fallback_indices.get(class_id, 0)
                fallback_indices[class_id] = fallback_index + 1
                track_id = (
                    int(box.id[0].item()) if box.id is not None else fallback_index
                )
                color_key = class_id, track_id
                bounds = x1, y1, x2, y2
                detections.append((color_key, class_id, bounds))
                if class_id == 0:
                    person_candidates.append(
                        (float(box.conf[0].item()), color_key, bounds)
                    )

        person_candidates.sort(key=lambda person: person[0], reverse=True)
        people = person_candidates[:MAX_GESTURE_PEOPLE]
        annotated_frame = frame.copy()
        self.frame_number += 1
        self._update_gender_labels(frame, people)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = time.monotonic_ns() // 1_000_000
        hand_result = self.hand_detector.detect_for_video(mp_frame, timestamp_ms)

        hands = hand_result.hand_landmarks
        frame_height, frame_width = frame.shape[:2]
        hands_by_person = {person[1]: [] for person in people}
        for hand in hands:
            if len(hand) < 21:
                continue

            person_key = self._person_for_hand(
                hand, people, frame_width, frame_height
            )
            if person_key is not None:
                hands_by_person[person_key].append(hand)

        current_person_keys = set(hands_by_person)
        self.gesture_states = {
            key: state
            for key, state in self.gesture_states.items()
            if key[0] in current_person_keys
        }
        self.gesture_confirmation_frames = {
            key: count
            for key, count in self.gesture_confirmation_frames.items()
            if key[0] in current_person_keys
        }

        for _, person_key, _ in people:
            person_hands = hands_by_person[person_key]
            heart_detected = self.heart_gesture(person_hands)
            heart_position = None
            if heart_detected:
                left, right = sorted(
                    person_hands[:2],
                    key=lambda hand: hand[0].x,
                )
                heart_position = (
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

            peace_hand = next(
                (hand for hand in person_hands if self.peace_gesture(hand)),
                None,
            )
            peace_detected = peace_hand is not None
            peace_position = None
            if peace_hand is not None:
                peace_position = (
                    int((peace_hand[8].x + peace_hand[12].x) * frame_width / 2),
                    int((peace_hand[8].y + peace_hand[12].y) * frame_height / 2),
                )

            rock_and_roll_hand = next(
                (hand for hand in person_hands if self.rock_and_roll_gesture(hand)),
                None,
            )
            rock_and_roll_position = None
            if rock_and_roll_hand is not None:
                rock_and_roll_position = (
                    int(
                        (rock_and_roll_hand[8].x + rock_and_roll_hand[20].x)
                        * frame_width
                        / 2
                    ),
                    int(
                        (rock_and_roll_hand[8].y + rock_and_roll_hand[20].y)
                        * frame_height
                        / 2
                    ),
                )

            shaka_hand = next(
                (hand for hand in person_hands if self.shaka_gesture(hand)),
                None,
            )
            shaka_position = None
            if shaka_hand is not None:
                shaka_position = (
                    int((shaka_hand[4].x + shaka_hand[20].x) * frame_width / 2),
                    int((shaka_hand[4].y + shaka_hand[20].y) * frame_height / 2),
                )

            for name, image, detected, position in (
                ('heart', self.heart_image, heart_detected, heart_position),
                ('peace', self.peace_image, peace_detected, peace_position),
                (
                    'rock_and_roll',
                    self.rock_and_roll_image,
                    rock_and_roll_hand is not None,
                    rock_and_roll_position,
                ),
                ('shaka', self.surf_boy_image, shaka_hand is not None, shaka_position),
            ):
                state_key = person_key, name
                active, started, display_position = self.gesture_states.get(
                    state_key, (False, None, None)
                )
                confirmation_frames = self._gesture_confirmation_count(
                    detected,
                    self.gesture_confirmation_frames.get(state_key, 0),
                )
                if confirmation_frames:
                    self.gesture_confirmation_frames[state_key] = confirmation_frames
                else:
                    self.gesture_confirmation_frames.pop(state_key, None)

                confirmed = confirmation_frames >= GESTURE_CONFIRMATION_FRAMES
                if confirmed:
                    if not active:
                        started = time.monotonic()
                    display_position = position
                active = confirmed

                if started is not None:
                    started = self._draw_gesture_overlay(
                        annotated_frame,
                        image,
                        display_position,
                        started,
                    )
                    if started is None:
                        display_position = None

                if not active and started is None:
                    self.gesture_states.pop(state_key, None)
                else:
                    self.gesture_states[state_key] = (
                        active,
                        started,
                        display_position,
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
                match = max(
                    (
                        (self.box_iou(bounds, previous[0]), key, previous)
                        for key, previous in self.track_states.items()
                        if key not in current_keys
                        and key not in matched_previous_keys
                        and key[0] == class_id
                    ),
                    key=lambda item: item[0],
                    default=None,
                )
                if match is not None and match[0] > TRACK_COLOR_REUSE_IOU:
                    _, previous_key, state = match
                    matched_previous_keys.add(previous_key)
                    self.colors[color_key] = state[1]
                    del self.track_states[previous_key]
                else:
                    color = self.colors.setdefault(
                        color_key,
                        BOX_COLORS[len(self.colors) % len(BOX_COLORS)],
                    )
                    state = bounds, color, self.frame_number - 1, 0, False

            _, color, last_seen, previous_hits, was_confirmed = state
            consecutive = last_seen == self.frame_number - 1
            hits = previous_hits + 1 if consecutive else 1
            confirmed = (
                was_confirmed and consecutive
                or hits >= YOLO_MIN_TRACK_HITS
            )
            self.track_states[color_key] = (
                bounds, color, self.frame_number, hits, confirmed
            )
            if confirmed:
                visible_detections.append((color_key, class_id, bounds, color))

        for color_key, class_id, (x1, y1, x2, y2), color in visible_detections:
            cv.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                BOX_THICKNESS,
            )

            if class_id == 0:
                label = self.gender_labels.get(color_key, (None,))[0] or PERSON_LABEL
                font = cv.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), baseline = cv.getTextSize(
                    label, font, PERSON_LABEL_FONT_SCALE, PERSON_LABEL_THICKNESS
                )
                label_width = text_width + PERSON_LABEL_PADDING * 2
                label_height = text_height + baseline + PERSON_LABEL_PADDING * 2
                frame_width = annotated_frame.shape[1]
                label_x = max(0, min(x1, frame_width - label_width))
                label_y = max(0, y1 - label_height)

                cv.rectangle(
                    annotated_frame,
                    (label_x, label_y),
                    (label_x + label_width, label_y + label_height),
                    color,
                    -1,
                )
                cv.putText(
                    annotated_frame,
                    label,
                    (
                        label_x + PERSON_LABEL_PADDING,
                        label_y + PERSON_LABEL_PADDING + text_height,
                    ),
                    font,
                    PERSON_LABEL_FONT_SCALE,
                    PERSON_LABEL_TEXT_COLOR,
                    PERSON_LABEL_THICKNESS,
                    cv.LINE_AA,
                )
        return annotated_frame