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
        # Candidate evidence is accumulated per current tracker ID. Automatic
        # promotion requires both zone entry and loitering evidence.
        self.candidates = {}
        # Track IDs are only meaningful inside the current process. Never use
        # the persisted last_track_id as an identity across a new run.
        self.session_bindings = {}

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

    def _target_by_id(self, target_id):
        row = self.connection.execute(
            "SELECT * FROM targets WHERE target_id = ?", (int(target_id),)
        ).fetchone()
        return self._row_to_target(row)

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
        target_id = self.session_bindings.get(int(track_id))
        target = self._target_by_id(target_id) if target_id is not None else None
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
            if face_feature is not None and target["face_embedding"] is not None:
                # When both faces are available, require the face match. This
                # prevents a similar outfit/body from hijacking a target.
                if face_distance is None or face_distance > self.face_threshold:
                    continue
                score = face_distance / max(self.face_threshold, 1e-6)
            elif body_distance is not None:
                score = body_distance / max(self.body_threshold, 1e-6)
            else:
                continue
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
            target_id = self.session_bindings.get(int(track_id))
            target = self._target_by_id(target_id) if target_id is not None else None
            if target is None:
                target = self._match_embeddings(profile.get("body"), profile.get("face"))
            if target is not None:
                self._update_target(target, track_id, profile)
                self.session_bindings[int(track_id)] = target["target_id"]
                bindings[int(track_id)] = target["target_id"]

        if manual_track_id is not None and manual_track_id in profiles:
            target_id = self.session_bindings.get(int(manual_track_id))
            target = self._target_by_id(target_id) if target_id is not None else None
            if target is None:
                target = self._create_target(
                    manual_track_id, profiles[manual_track_id], "manual_selection", manual_label
                )
            self.session_bindings[int(manual_track_id)] = target["target_id"]
            bindings[int(manual_track_id)] = target["target_id"]

        for event in events:
            track_id = int(event["track_id"])
            target_id = bindings.get(track_id)
            if target_id is not None:
                self._record_event(target_id, event)
            else:
                track_id = int(track_id)
                evidence = self.candidates.setdefault(
                    track_id, {"intrusions": 0, "loitering": 0}
                )
                if event["event"] == "restricted_area_intrusion":
                    evidence["intrusions"] += 1
                elif event["event"] == "loitering":
                    evidence["loitering"] += 1
                if (
                    evidence["intrusions"] >= self.min_intrusions
                    and evidence["loitering"] >= self.min_loitering
                    and track_id in profiles
                ):
                    target = self._create_target(
                        track_id, profiles[track_id], f"auto_{event['event']}",
                    )
                    self.session_bindings[int(track_id)] = target["target_id"]
                    bindings[track_id] = target["target_id"]
                    self._record_event(target["target_id"], event)

        return {
            track_id: {"target_id": target_id, "label": f"WATCHLIST-{target_id}"}
            for track_id, target_id in bindings.items()
        }
