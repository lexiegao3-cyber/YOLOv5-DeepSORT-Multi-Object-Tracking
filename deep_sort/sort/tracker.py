# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track, TrackState


class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """
    GATING_THRESHOLD = np.sqrt(kalman_filter.chi2inv95[4])

    def __init__(self, metric, max_iou_distance=0.9, max_age=30, n_init=3, _lambda=0,
                 face_metric=None, face_weight=0.45, lost_track_ttl=900):
        self.metric = metric
        self.face_metric = face_metric
        self.face_weight = face_weight
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.lost_track_ttl = lost_track_ttl
        self.lost_tracks = []

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def increment_ages(self):
        self._advance_lost_tracks()
        for track in self.tracks:
            track.increment_age()
            if track.time_since_update > self.max_age:
                self._store_lost_track(track)
            track.mark_missed()
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def update(self, detections, classes):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        self._advance_lost_tracks()

        # Run matching cascade for active tracks.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx], classes[detection_idx])
        for track_idx in unmatched_tracks:
            if track_idx < len(self.tracks) and self.tracks[track_idx].time_since_update > self.max_age:
                self._store_lost_track(self.tracks[track_idx])
            self.tracks[track_idx].mark_missed()
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Reactivate deleted tracks before assigning new IDs.
        lost_matches, unmatched_detections = self._match_lost_tracks(
            detections, unmatched_detections
        )
        for lost_idx, detection_idx in lost_matches:
            self._reactivate_track(
                self.lost_tracks[lost_idx],
                detections[detection_idx],
                classes[detection_idx],
            )
        reactivated_indices = {lost_idx for lost_idx, _ in lost_matches}
        if reactivated_indices:
            self.lost_tracks = [
                record for index, record in enumerate(self.lost_tracks)
                if index not in reactivated_indices
            ]

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], classes[detection_idx].item())

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        face_features, face_targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
            if self.face_metric is not None:
                face_features += track.face_features
                face_targets += [track.track_id for _ in track.face_features]
                track.face_features = []
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)
        if self.face_metric is not None:
            self.face_metric.partial_fit(
                np.asarray(face_features), np.asarray(face_targets), active_targets
            )

    def _advance_lost_tracks(self):
        """Age and prune tracks that have already exceeded MAX_AGE."""
        for record in self.lost_tracks:
            record['lost_frames'] += 1
        self.lost_tracks = [
            record for record in self.lost_tracks
            if record['lost_frames'] <= self.lost_track_ttl
        ]

    def _store_lost_track(self, track):
        """Save a confirmed track's galleries before removing it from active tracks."""
        if track.state != TrackState.Confirmed:
            return
        if any(record['track_id'] == track.track_id for record in self.lost_tracks):
            return
        self.lost_tracks.append({
            'track_id': track.track_id,
            'class_id': track.class_id,
            'hits': track.hits,
            'lost_frames': 0,
            'features': [
                np.asarray(feature, dtype=np.float32).copy()
                for feature in self.metric.samples.get(track.track_id, track.features)
            ],
            'face_features': [
                np.asarray(feature, dtype=np.float32).copy()
                for feature in (
                    self.face_metric.samples.get(track.track_id, track.face_features)
                    if self.face_metric is not None else []
                )
            ],
        })

    def _lost_cost_metric(self, records, dets, record_indices, detection_indices):
        """Calculate appearance-only costs for reactivating lost tracks."""
        cost_matrix = np.full(
            (len(record_indices), len(detection_indices)), linear_assignment.INFTY_COST,
            dtype=np.float32,
        )
        for row, record_idx in enumerate(record_indices):
            record = records[record_idx]
            body_gallery = record['features']
            if not body_gallery:
                continue
            for col, detection_idx in enumerate(detection_indices):
                detection = dets[detection_idx]
                body_cost = float(
                    self.metric._metric(
                        np.asarray(body_gallery),
                        np.asarray([detection.feature], dtype=np.float32),
                    )[0]
                )
                cost = body_cost
                face_feature = detection.face_feature
                face_gallery = record['face_features']
                if self.face_metric is not None and face_feature is not None and face_gallery:
                    face_cost = float(
                        self.face_metric._metric(
                            np.asarray(face_gallery),
                            np.asarray([face_feature], dtype=np.float32),
                        )[0]
                    )
                    cost = (
                        (1 - self.face_weight) * body_cost
                        + self.face_weight * face_cost
                    )
                    body_bad = body_cost > self.metric.matching_threshold
                    face_bad = face_cost > self.face_metric.matching_threshold
                    if body_bad and face_bad:
                        continue
                elif body_cost > self.metric.matching_threshold:
                    continue
                cost_matrix[row, col] = cost
        return cost_matrix

    def _match_lost_tracks(self, detections, detection_indices):
        if not self.lost_tracks or not detection_indices:
            return [], detection_indices
        candidate_indices = list(range(len(self.lost_tracks)))
        matches, _, unmatched_detections = linear_assignment.min_cost_matching(
            self._lost_cost_metric,
            linear_assignment.INFTY_COST - 1,
            self.lost_tracks,
            detections,
            candidate_indices,
            detection_indices,
        )
        return matches, unmatched_detections

    def _reactivate_track(self, record, detection, class_id):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track = Track(
            mean, covariance, record['track_id'], class_id, self.n_init, self.max_age,
            detection.feature, detection.face_feature,
        )
        track.state = TrackState.Confirmed
        track.hits = max(record['hits'] + 1, self.n_init)
        track.features = record['features'] + [detection.feature]
        track.face_features = record['face_features'][:]
        if detection.face_feature is not None:
            track.face_features.append(detection.face_feature)
        self.tracks.append(track)

    def _face_cost_metric(self, tracks, dets, track_indices, detection_indices):
        """Return face costs where both the track and detection have a face feature."""
        cost_matrix = np.full(
            (len(track_indices), len(detection_indices)), np.nan, dtype=np.float32
        )
        if self.face_metric is None:
            return cost_matrix

        track_rows = []
        target_ids = []
        for row, track_idx in enumerate(track_indices):
            track_id = tracks[track_idx].track_id
            if track_id in self.face_metric.samples:
                track_rows.append(row)
                target_ids.append(track_id)

        detection_cols = []
        face_features = []
        for col, detection_idx in enumerate(detection_indices):
            face_feature = dets[detection_idx].face_feature
            if face_feature is not None and np.asarray(face_feature).size > 0:
                detection_cols.append(col)
                face_features.append(face_feature)

        if not track_rows or not detection_cols:
            return cost_matrix

        face_cost = self.face_metric.distance(
            np.asarray(face_features, dtype=np.float32), np.asarray(target_ids)
        )
        cost_matrix[np.ix_(track_rows, detection_cols)] = face_cost
        return cost_matrix

    def _full_cost_metric(self, tracks, dets, track_indices, detection_indices):
        """
        This implements the full lambda-based cost-metric. However, in doing so, it disregards
        the possibility to gate the position only which is provided by
        linear_assignment.gate_cost_matrix(). Instead, I gate by everything.
        Note that the Mahalanobis distance is itself an unnormalised metric. Given the cosine
        distance being normalised, we employ a quick and dirty normalisation based on the
        threshold: that is, we divide the positional-cost by the gating threshold, thus ensuring
        that the valid values range 0-1.
        Note also that the authors work with the squared distance. I also sqrt this, so that it
        is more intuitive in terms of values.
        """
        # Compute First the Position-based Cost Matrix
        pos_cost = np.empty([len(track_indices), len(detection_indices)])
        msrs = np.asarray([dets[i].to_xyah() for i in detection_indices])
        for row, track_idx in enumerate(track_indices):
            pos_cost[row, :] = np.sqrt(
                self.kf.gating_distance(
                    tracks[track_idx].mean, tracks[track_idx].covariance, msrs, False
                )
            ) / self.GATING_THRESHOLD
        pos_gate = pos_cost > 1.0
        # Now Compute the Appearance-based Cost Matrix
        app_cost = self.metric.distance(
            np.array([dets[i].feature for i in detection_indices]),
            np.array([tracks[i].track_id for i in track_indices]),
        )
        app_gate = app_cost > self.metric.matching_threshold
        face_cost = self._face_cost_metric(
            tracks, dets, track_indices, detection_indices
        )
        face_available = np.isfinite(face_cost)
        face_gate = np.zeros_like(face_available)
        if self.face_metric is not None and np.any(face_available):
            face_gate[face_available] = (
                face_cost[face_available] > self.face_metric.matching_threshold
            )
        # Combine motion and appearance costs
        cost_matrix = self._lambda * pos_cost + (1 - self._lambda) * app_cost
        if np.any(face_available):
            cost_matrix[face_available] = (
                (1 - self.face_weight) * app_cost[face_available]
                + self.face_weight * face_cost[face_available]
            )
        # Age-adaptive gating:
        # For recently observed tracks, use both motion and appearance gating.
        # For stale tracks, Kalman prediction may have drifted, so rely mainly
        # on appearance features for re-association.
        for row, track_idx in enumerate(track_indices):
            time_lost = tracks[track_idx].time_since_update
            invalid = app_gate[row].copy()
            # When a reliable face exists, either body or face appearance may
            # rescue the other modality; missing faces fall back to body ReID.
            invalid[face_available[row]] &= face_gate[row][face_available[row]]
            if time_lost <= 10:
                # Recent track: require both reasonable position and appearance.
                invalid |= pos_gate[row]
            # Stale track: ignore unreliable position gate and use appearance.
            cost_matrix[row, invalid] = linear_assignment.INFTY_COST
        # Return Matrix
        return cost_matrix

    def _match(self, detections):
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            self._full_cost_metric,
            linear_assignment.INFTY_COST - 1,  # no need for self.metric.matching_threshold here,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, class_id):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, class_id, self.n_init, self.max_age,
            detection.feature, detection.face_feature))
        self._next_id += 1
