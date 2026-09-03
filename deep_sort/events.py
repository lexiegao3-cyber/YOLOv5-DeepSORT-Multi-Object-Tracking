"""Restricted-area and loitering event detection for tracked objects.

The event layer deliberately consumes tracker output instead of detector
internals.  This keeps it compatible with the existing DeepSORT ID persistence
and makes the same code usable for webcam, video, image folders and MOT data.
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np


def _value(mapping, key, default=None):
    """Read a key from either a dict or an EasyDict-like object."""
    if mapping is None:
        return default
    if hasattr(mapping, "get"):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


class PolygonZone:
    """A named polygon in image or calibrated ground-plane coordinates."""

    def __init__(self, name, polygon, color=(0, 0, 255), coordinate_space="image", mapper=None):
        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("A zone polygon must contain at least three [x, y] points")
        self.name = str(name)
        self.polygon = points.reshape((-1, 1, 2))
        self.color = tuple(int(channel) for channel in color)
        self.coordinate_space = str(coordinate_space).lower()
        self.mapper = mapper
        if self.coordinate_space == "ground_plane" and mapper is None:
            raise ValueError("ground_plane zones require camera calibration")

    def contains(self, point):
        if self.coordinate_space == "ground_plane":
            point = self.mapper.image_to_world(point)
        return cv2.pointPolygonTest(
            self.polygon, (float(point[0]), float(point[1])), False
        ) >= 0

    def image_polygon(self):
        if self.coordinate_space == "ground_plane":
            points = self.mapper.world_to_image(self.polygon[:, 0, :])
            return np.rint(points).astype(np.int32).reshape((-1, 1, 2))
        return np.rint(self.polygon).astype(np.int32)

    def draw(self, frame, alpha=0.18):
        polygon = self.image_polygon()
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], self.color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [polygon], True, self.color, 2, cv2.LINE_AA)
        anchor = tuple(int(value) for value in polygon[0, 0])
        cv2.putText(
            frame, self.name, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            self.color, 2, cv2.LINE_AA,
        )


class PerspectiveMapper:
    """Map image pixels to a calibrated ground-plane coordinate system."""

    def __init__(self, image_points, world_points):
        image_points = np.asarray(image_points, dtype=np.float32)
        world_points = np.asarray(world_points, dtype=np.float32)
        if image_points.shape != (4, 2) or world_points.shape != (4, 2):
            raise ValueError("Calibration requires exactly four image and world points")
        self.image_to_world_matrix = cv2.getPerspectiveTransform(image_points, world_points)
        self.world_to_image_matrix = cv2.getPerspectiveTransform(world_points, image_points)

    @staticmethod
    def _transform(points, matrix):
        points = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
        return cv2.perspectiveTransform(points, matrix).reshape((-1, 2))

    def image_to_world(self, point):
        return self._transform([point], self.image_to_world_matrix)[0]

    def world_to_image(self, points):
        return self._transform(points, self.world_to_image_matrix)


class EventDetector:
    """Track-aware restricted-area intrusion and loitering detector.

    ``outputs`` must contain rows in the existing DeepSort format:
    ``[x1, y1, x2, y2, track_id, class_id]``.
    """

    def __init__(
        self,
        zones=None,
        loitering_enabled=True,
        loitering_duration=10.0,
        loitering_max_movement=60.0,
        loitering_max_step_movement=0.08,
        loitering_require_zone=True,
        history_size=120,
        track_ttl=90,
        use_footpoint=False,
        trajectory_max_gap=15,
        trajectory_max_jump=120.0,
        coordinate_mapper=None,
    ):
        self.zones = list(zones or [])
        self.loitering_enabled = bool(loitering_enabled)
        self.loitering_duration = float(loitering_duration)
        self.loitering_max_movement = float(loitering_max_movement)
        self.loitering_max_step_movement = max(float(loitering_max_step_movement), 0.0)
        self.loitering_require_zone = bool(loitering_require_zone)
        self.history_size = max(int(history_size), 2)
        self.track_ttl = max(int(track_ttl), 1)
        self.use_footpoint = bool(use_footpoint)
        self.trajectory_max_gap = max(int(trajectory_max_gap), 1)
        self.trajectory_max_jump = max(float(trajectory_max_jump), 1.0)
        self.coordinate_mapper = coordinate_mapper
        self.track_states = {}

    def update(self, outputs, frame_idx, fps=15.0):
        """Update event state and return ``(events, visible_statuses)``."""
        fps = max(float(fps), 1e-6)
        current_frame = int(frame_idx)
        events = []
        statuses = {}

        if outputs is None:
            outputs = []
        for output in outputs:
            if len(output) < 6:
                continue
            x1, y1, x2, y2 = [float(value) for value in output[:4]]
            track_id = int(output[4])
            class_id = int(output[5])
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            bottom_center = ((x1 + x2) / 2.0, y2)
            event_point = bottom_center if self.use_footpoint else center
            state = self.track_states.setdefault(
                track_id,
                {
                    "history": deque(maxlen=self.history_size),
                    "inside_zones": set(),
                    "anchor": None,
                    "stable_since": None,
                    "loitering_alerted": False,
                    "previous_motion_point": None,
                    "previous_motion_frame": None,
                },
            )
            state["history"].append((current_frame, event_point))
            state["last_frame"] = current_frame
            state["class_id"] = class_id

            inside_zones = {
                zone.name for zone in self.zones if zone.contains(event_point)
            }
            previously_inside = state["inside_zones"]
            for zone_name in sorted(inside_zones - previously_inside):
                events.append(self._event(
                    "restricted_area_intrusion", track_id, class_id,
                    current_frame, current_frame / fps, zone_name, event_point,
                ))
            state["inside_zones"] = inside_zones

            eligible_for_loitering = (
                bool(inside_zones) if self.loitering_require_zone else True
            )
            if not self.loitering_enabled or not eligible_for_loitering:
                state["anchor"] = None
                state["stable_since"] = None
                state["loitering_alerted"] = False
            else:
                motion_point = (
                    self.coordinate_mapper.image_to_world(event_point)
                    if self.coordinate_mapper is not None else event_point
                )
                point = np.asarray(motion_point, dtype=np.float32)
                previous_point = state["previous_motion_point"]
                previous_frame = state["previous_motion_frame"]
                step_distance = None
                if previous_point is not None and previous_frame is not None:
                    frame_gap = max(current_frame - previous_frame, 1)
                    step_distance = np.linalg.norm(point - previous_point) / frame_gap
                state["previous_motion_point"] = point
                state["previous_motion_frame"] = current_frame
                anchor = state["anchor"]
                # A person who keeps walking through the zone must not be
                # classified as loitering just because the path curves. Start
                # the dwell timer only after per-frame movement drops below
                # the configured stop threshold.
                is_moving = (
                    step_distance is not None
                    and step_distance > self.loitering_max_step_movement
                )
                if (
                    is_moving
                    or anchor is None
                    or np.linalg.norm(point - anchor) > self.loitering_max_movement
                ):
                    state["anchor"] = point
                    state["stable_since"] = current_frame
                    state["loitering_alerted"] = False
                stable_since = state["stable_since"]
                if (
                    stable_since is not None
                    and not state["loitering_alerted"]
                    and (current_frame - stable_since) / fps >= self.loitering_duration
                ):
                    zone_name = sorted(inside_zones)[0] if inside_zones else None
                    events.append(self._event(
                        "loitering", track_id, class_id, current_frame,
                        current_frame / fps, zone_name, event_point,
                    ))
                    state["loitering_alerted"] = True

            statuses[track_id] = {
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": center,
                "event_point": event_point,
                "inside_zones": sorted(inside_zones),
                "loitering": bool(state["loitering_alerted"]),
                "history": list(state["history"]),
            }

        stale_ids = [
            track_id for track_id, state in self.track_states.items()
            if current_frame - state.get("last_frame", current_frame) > self.track_ttl
        ]
        for track_id in stale_ids:
            del self.track_states[track_id]
        return events, statuses

    @staticmethod
    def _event(event_type, track_id, class_id, frame, timestamp, zone, center):
        return {
            "event": event_type,
            "track_id": int(track_id),
            "class_id": int(class_id),
            "frame": int(frame),
            "timestamp_seconds": round(float(timestamp), 3),
            "zone": zone,
            "center": [round(float(center[0]), 2), round(float(center[1]), 2)],
        }

    def draw(self, frame, statuses, events=None, frame_idx=None, watchlist_statuses=None):
        """Draw zones, trajectories and current event labels on ``frame``."""
        for zone in self.zones:
            zone.draw(frame)

        watchlist_filter = watchlist_statuses is not None
        watchlist_statuses = watchlist_statuses or {}
        current_frame = int(frame_idx) if frame_idx is not None else None
        for track_id, state in self.track_states.items():
            if watchlist_filter and track_id not in watchlist_statuses:
                continue
            if current_frame is not None and current_frame - state.get("last_frame", current_frame) > self.track_ttl:
                continue
            points = list(state["history"])
            segment = []
            for index, item in enumerate(points):
                if index and not self._trajectory_link_allowed(points[index - 1], item):
                    self._draw_trajectory_segment(
                        frame, segment, track_id,
                        (0, 0, 255) if track_id in watchlist_statuses else (0, 255, 0),
                    )
                    segment = []
                segment.append(item[1])
            self._draw_trajectory_segment(
                frame, segment, track_id,
                (0, 0, 255) if track_id in watchlist_statuses else (0, 255, 0),
            )

        for track_id, status in statuses.items():
            if watchlist_filter and track_id not in watchlist_statuses:
                continue
            if status["inside_zones"]:
                label = f"WATCHLIST ID {track_id} IN ZONE"
                color = (0, 0, 255)
                if status["loitering"]:
                    label = f"WATCHLIST ID {track_id} LOITERING"
                    color = (0, 0, 255)
                x1, y1, _, _ = status["bbox"]
                cv2.putText(frame, label, (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        if events:
            y = 30
            for event in events:
                if watchlist_filter and int(event["track_id"]) not in watchlist_statuses:
                    continue
                label = f"ALERT: {event['event'].upper()} - ID {event['track_id']}"
                cv2.putText(frame, label, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
                y += 28
        return frame

    def _trajectory_link_allowed(self, previous, current):
        previous_frame, previous_point = previous
        current_frame, current_point = current
        gap = current_frame - previous_frame
        if gap < 1 or gap > self.trajectory_max_gap:
            return False
        distance = np.linalg.norm(np.asarray(current_point) - np.asarray(previous_point))
        return distance <= self.trajectory_max_jump * gap

    def _draw_trajectory_segment(self, frame, points, track_id, color):
        if len(points) < 2:
            return
        history = np.asarray(points, dtype=np.int32)
        cv2.polylines(frame, [history.reshape((-1, 1, 2))], False,
                      color, 2, cv2.LINE_AA)


def load_event_config(config):
    """Build an ``EventDetector`` from the EVENTS YAML mapping."""
    events_cfg = _value(config, "EVENTS", {})
    coordinate_space = str(_value(events_cfg, "COORDINATE_SYSTEM", "image")).lower()
    mapper = None
    if coordinate_space == "ground_plane":
        calibration_cfg = _value(events_cfg, "CALIBRATION", {})
        mapper = PerspectiveMapper(
            _value(calibration_cfg, "IMAGE_POINTS", []),
            _value(calibration_cfg, "WORLD_POINTS", []),
        )
    zone_configs = _value(events_cfg, "ZONES", []) or []
    zones = []
    for index, zone_cfg in enumerate(zone_configs):
        name = _value(zone_cfg, "NAME", f"restricted_zone_{index + 1}")
        polygon = _value(zone_cfg, "POLYGON", [])
        color = _value(zone_cfg, "COLOR", [0, 0, 255])
        zones.append(PolygonZone(name, polygon, color, coordinate_space, mapper))

    loiter_cfg = _value(events_cfg, "LOITERING", {})
    return EventDetector(
        zones=zones,
        loitering_enabled=_value(loiter_cfg, "ENABLED", True),
        loitering_duration=_value(loiter_cfg, "MIN_DURATION_SECONDS", 10.0),
        loitering_max_movement=_value(
            loiter_cfg, "MAX_MOVEMENT_UNITS",
            _value(loiter_cfg, "MAX_MOVEMENT_PIXELS", 60.0),
        ),
        loitering_max_step_movement=_value(
            loiter_cfg, "MAX_STEP_MOVEMENT_UNITS", 0.08
        ),
        loitering_require_zone=_value(loiter_cfg, "REQUIRE_RESTRICTED_ZONE", True),
        history_size=_value(events_cfg, "TRAJECTORY_HISTORY_FRAMES", 120),
        track_ttl=_value(events_cfg, "TRACK_STATE_TTL_FRAMES", 90),
        use_footpoint=_value(events_cfg, "USE_FOOTPOINT", coordinate_space == "ground_plane"),
        trajectory_max_gap=_value(events_cfg, "TRAJECTORY_MAX_GAP_FRAMES", 15),
        trajectory_max_jump=_value(events_cfg, "TRAJECTORY_MAX_JUMP_PIXELS_PER_FRAME", 120),
        coordinate_mapper=mapper,
    )


class EventLogger:
    """Append one JSON object per event, if a log path is configured."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, events):
        if not self.path or not events:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
