import numpy as np
import torch
import cv2
import sys
import gdown
from os.path import exists as file_exists, join
from insightface.app import FaceAnalysis

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker
from .deep.reid_model_factory import show_downloadeable_models, get_model_link, is_model_in_factory, \
    is_model_type_in_model_path, get_model_type, show_supported_models

sys.path.append('deep_sort/deep/reid')
from torchreid.utils import FeatureExtractor
from torchreid.utils.tools import download_url

show_downloadeable_models()

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model, device, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3,
                 nn_budget=100, face_model='buffalo_l', face_max_dist=0.4,
                 face_weight=0.45, face_min_score=0.65, face_min_size=40,
                 lost_track_ttl=900):
        # models trained on: market1501, dukemtmcreid and msmt17
        if is_model_in_factory(model):
            # download the model
            model_path = join('deep_sort/deep/checkpoint', model + '.pth')
            if not file_exists(model_path):
                gdown.download(get_model_link(model), model_path, quiet=False)

            self.extractor = FeatureExtractor(
                # get rid of dataset information DeepSort model name
                model_name=model.rsplit('_', 1)[:-1][0],
                model_path=model_path,
                device=str(device)
            )
        else:
            if is_model_type_in_model_path(model):
                model_name = get_model_type(model)
                self.extractor = FeatureExtractor(
                    model_name=model_name,
                    model_path=model,
                    device=str(device)
                )
            else:
                print('Cannot infere model name from provided DeepSort path, should be one of the following:')
                show_supported_models()
                exit()

        self.face_min_score = face_min_score
        self.face_min_size = face_min_size
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if str(device).startswith('cuda') else ['CPUExecutionProvider']
        )
        self.face_app = FaceAnalysis(
            name=face_model,
            allowed_modules=['detection', 'recognition'],
            providers=providers,
        )
        self.face_app.prepare(
            ctx_id=0 if str(device).startswith('cuda') else -1,
            det_size=(640, 640),
        )

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        face_metric = NearestNeighborDistanceMetric(
            "cosine", face_max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init,
            face_metric=face_metric, face_weight=face_weight,
            lost_track_ttl=lost_track_ttl)

    def update(self, bbox_xywh, confidences, classes, ori_img, use_yolo_preds=False):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        face_features = self._get_face_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(bbox_tlwh[i], conf, features[i], face_features[i])
            for i, conf in enumerate(confidences)
        ]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if use_yolo_preds:
                det = track.get_yolo_pred()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh)
            else:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            if im.size == 0:
                im = np.zeros((1, 1, 3), dtype=ori_img.dtype)
            else:
                # OpenCV frames are BGR; Torchreid expects RGB input.
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def _get_face_features(self, bbox_xywh, ori_img):
        """Extract one quality-filtered face embedding for each person box."""
        face_features = [None] * len(bbox_xywh)
        if len(bbox_xywh) == 0:
            return face_features

        faces = self.face_app.get(ori_img)
        if not faces:
            return face_features

        if isinstance(bbox_xywh, torch.Tensor):
            boxes = bbox_xywh.detach().cpu().numpy()
        else:
            boxes = np.asarray(bbox_xywh)

        for person_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            best_face = None
            best_score = -1.0
            for face in faces:
                face_box = np.asarray(face.bbox, dtype=float)
                face_width, face_height = face_box[2] - face_box[0], face_box[3] - face_box[1]
                if min(face_width, face_height) < self.face_min_size:
                    continue
                if float(getattr(face, 'det_score', 0.0)) < self.face_min_score:
                    continue
                center_x = (face_box[0] + face_box[2]) / 2.0
                center_y = (face_box[1] + face_box[3]) / 2.0
                if not (x1 <= center_x <= x2 and y1 <= center_y <= y2):
                    continue
                score = float(getattr(face, 'det_score', 0.0)) * face_width * face_height
                if score > best_score:
                    best_face, best_score = face, score

            if best_face is None:
                continue
            embedding = getattr(best_face, 'normed_embedding', None)
            if embedding is None:
                embedding = np.asarray(best_face.embedding, dtype=np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            face_features[person_idx] = np.asarray(embedding, dtype=np.float32)

        return face_features
