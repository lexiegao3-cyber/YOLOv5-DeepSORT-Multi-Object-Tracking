"""Persistent watchlist identities backed by a local SQLite database."""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from pathlib import Path

import numpy as np


def _normalise(feature):
    if feature is None:
        return None
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else None


def _cosine_distance(left, right):
    left = _normalise(left)
    right = _normalise(right)
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(1.0 - np.dot(left, right))


def _now():
    return datetime.now(timezone.utc).isoformat()


class WatchlistDB:
    """Store persistent target profiles and bind them to current track IDs."""

    def __init__(self, path, body_threshold=0.35, face_threshold=0.40,
                 min_intrusions=3, min_loitering=1):
        self.path = str(path)
        self.body_threshold = float(body_threshold)
        self.face_threshold = float(face_threshold)
        self.min_intrusions = max(int(min_intrusions), 1)
        self.min_loitering = max(int(min_loitering), 1)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS targets (
                target_id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                reason TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                last_track_id INTEGER,
                intrusion_count INTEGER NOT NULL DEFAULT 0,
                loitering_count INTEGER NOT NULL DEFAULT 0,
                body_embedding BLOB,
                face_embedding BLOB
            );
            CREATE TABLE IF NOT EXISTS target_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id INTEGER NOT NULL,
                event TEXT NOT NULL,
                frame INTEGER,
                timestamp_seconds REAL,
                zone TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(target_id) REFERENCES targets(target_id)
            );
            """
        )
        self.connection.commit()
        self.candidates = {}

    @staticmethod
    def _profile_blob(feature):
        feature = _normalise(feature)
        return feature.tobytes() if feature is not None else None

    @staticmethod
    def _profile_from_blob(blob):
        return np.frombuffer(blob, dtype=np.float32).copy() if blob else None

    def _row_to_target(self, row):
        if row is None:
            return None
        return {
            "target_id": int(row["target_id"]),
            "label": row["label"],
            "reason": row["reason"],
            "last_track_id": row["last_track_id"],
            "intrusion_count": int(row["intrusion_count"]),
            "loitering_count": int(row["loitering_count"]),
            "body_embedding": self._profile_from_blob(row["body_embedding"]),
            "face_embedding": self._profile_from_blob(row["face_embedding"]),
        }

    def _targets(self):
        return [self._row_to_target(row) for row in self.connection.execute(
            "SELECT * FROM targets ORDER BY target_id"
        )]

    def _find_by_track_id(self, track_id):
        row = self.connection.execute(
            "SELECT * FROM targets WHERE last_track_id = ? ORDER BY target_id LIMIT 1",
            (int(track_id),),
        ).fetchone()
        return self._row_to_target(row)

    def status_for_track(self, track_id):
        """Return the persistent watchlist label currently bound to a track."""
        target = self._find_by_track_id(track_id)
        if target is None:
            return None
        return {"target_id": target["target_id"], "label": f"WATCHLIST-{target['target_id']}"}

    def _match_embeddings(self, body_feature, face_feature):
        body_feature = _normalise(body_feature)
        face_feature = _normalise(face_feature)
        best_target, best_score = None, float("inf")
        for target in self._targets():
            body_distance = _cosine_distance(body_feature, target["body_embedding"])
            face_distance = _cosine_distance(face_feature, target["face_embedding"])
            normalised_scores = []
            if body_distance is not None:
                normalised_scores.append(body_distance / max(self.body_threshold, 1e-6))
            if face_distance is not None:
                normalised_scores.append(face_distance / max(self.face_threshold, 1e-6))
            if not normalised_scores:
                continue
            # Face is more discriminative when available; a good face or body
            # match can recover a target after a long occlusion.
            score = min(normalised_scores)
            if score <= 1.0 and score < best_score:
                best_target, best_score = target, score
        return best_target

    def _create_target(self, track_id, profile, reason, label=None):
        timestamp = _now()
        label = label or f"suspect_{track_id}"
        cursor = self.connection.execute(
            """
            INSERT INTO targets
            (label, reason, first_seen, last_seen, last_track_id,
             body_embedding, face_embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                label, reason, timestamp, timestamp, int(track_id),
                self._profile_blob(profile.get("body")),
                self._profile_blob(profile.get("face")),
            ),
        )
        self.connection.commit()
        return self._row_to_target(self.connection.execute(
            "SELECT * FROM targets WHERE target_id = ?", (cursor.lastrowid,)
        ).fetchone())

    def _update_target(self, target, track_id, profile):
        body = profile.get("body") if profile.get("body") is not None else target["body_embedding"]
        face = profile.get("face") if profile.get("face") is not None else target["face_embedding"]
        self.connection.execute(
            """
            UPDATE targets SET last_seen = ?, last_track_id = ?,
              body_embedding = ?, face_embedding = ? WHERE target_id = ?
            """,
            (
                _now(), int(track_id), self._profile_blob(body),
                self._profile_blob(face), target["target_id"],
            ),
        )
        self.connection.commit()

    def _record_event(self, target_id, event):
        event_type = event["event"]
        self.connection.execute(
            """
            INSERT INTO target_events
            (target_id, event, frame, timestamp_seconds, zone, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                target_id, event_type, event.get("frame"),
                event.get("timestamp_seconds"), event.get("zone"), _now(),
            ),
        )
        column = "intrusion_count" if event_type == "restricted_area_intrusion" else "loitering_count"
        self.connection.execute(
            f"UPDATE targets SET {column} = {column} + 1 WHERE target_id = ?",
            (target_id,),
        )
        self.connection.commit()

    def observe(self, profiles, events=None, manual_track_id=None, manual_label="suspect"):
        """Bind current tracks to persistent targets and record security events.

        ``profiles`` maps tracker IDs to ``{"body": ndarray, "face": ndarray}``.
        The return value maps current tracker IDs to watchlist target metadata.
        """
        bindings = {}
        events = events or []
        for track_id, profile in profiles.items():
            target = self._find_by_track_id(track_id)
            if target is None:
                target = self._match_embeddings(profile.get("body"), profile.get("face"))
            if target is not None:
                self._update_target(target, track_id, profile)
                bindings[int(track_id)] = target["target_id"]

        if manual_track_id is not None and manual_track_id in profiles:
            target = self._find_by_track_id(manual_track_id)
            if target is None:
                target = self._create_target(
                    manual_track_id, profiles[manual_track_id], "manual_selection", manual_label
                )
            bindings[int(manual_track_id)] = target["target_id"]

        for event in events:
            track_id = int(event["track_id"])
            target_id = bindings.get(track_id)
            if target_id is not None:
                self._record_event(target_id, event)
            else:
                key = (track_id, event["event"])
                self.candidates[key] = self.candidates.get(key, 0) + 1
                threshold = (
                    self.min_intrusions if event["event"] == "restricted_area_intrusion"
                    else self.min_loitering
                )
                if self.candidates[key] >= threshold and track_id in profiles:
                    target = self._create_target(
                        track_id, profiles[track_id], f"auto_{event['event']}",
                    )
                    bindings[track_id] = target["target_id"]
                    self._record_event(target["target_id"], event)

        return {
            track_id: {"target_id": target_id, "label": f"WATCHLIST-{target_id}"}
            for track_id, target_id in bindings.items()
        }
