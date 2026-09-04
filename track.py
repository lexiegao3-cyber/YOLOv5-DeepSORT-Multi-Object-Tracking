# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.events import EventLogger, load_event_config
from deep_sort.ucf_crime import resolve_ucf_source
from deep_sort.watchlist import WatchlistDB

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def resolve_mot_image_dir(root, dataset, split, sequence):
    """Resolve standard MOT17/MOT20/<split>/<sequence>/img1 input."""
    root = Path(root).expanduser().resolve()
    candidates = [root / dataset / split, root / split]
    if root.name.lower() == split.lower():
        candidates.insert(0, root)
    for split_root in candidates:
        image_dir = split_root / sequence / 'img1'
        if image_dir.is_dir():
            return image_dir
    expected = [str(path / sequence / 'img1') for path in candidates]
    raise FileNotFoundError('MOT sequence not found. Expected: ' + ', '.join(expected))


def read_mot_fps(image_dir):
    """Read frameRate from the sequence metadata when available."""
    seqinfo = Path(image_dir).parent / 'seqinfo.ini'
    if not seqinfo.is_file():
        return None
    for line in seqinfo.read_text(encoding='utf-8').splitlines():
        key, separator, value = line.partition('=')
        if separator and key.strip().lower() == 'framerate':
            try:
                return float(value.strip())
            except ValueError:
                return None
    return None


def load_mot_detections(path, min_confidence=0.0):
    """Load MOT ``det/det.txt`` boxes keyed by 1-based frame number."""
    detections = {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'MOT detection file not found: {path}')
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.strip().split(',')
            if len(fields) < 7:
                raise ValueError(f'Invalid MOT detection at {path}:{line_number}')
            frame = int(float(fields[0]))
            x, y, width, height = [float(value) for value in fields[2:6]]
            confidence = float(fields[6])
            if confidence < min_confidence or width <= 0 or height <= 0:
                continue
            detections.setdefault(frame, []).append(
                [x, y, x + width, y + height, confidence, 0.0]
            )
    return detections


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
        project, exist_ok, update, save_crop = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    mot_detections = None
    if opt.mot_use_det:
        mot_det_path = Path(source).parent / 'det' / 'det.txt'
        mot_detections = load_mot_detections(mot_det_path, opt.mot_det_conf)
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                face_model=cfg.DEEPSORT.FACE_MODEL,
                face_max_dist=cfg.DEEPSORT.FACE_MAX_DIST,
                face_weight=cfg.DEEPSORT.FACE_WEIGHT,
                face_min_score=cfg.DEEPSORT.FACE_MIN_SCORE,
                face_min_size=cfg.DEEPSORT.FACE_MIN_SIZE,
                lost_track_ttl=cfg.DEEPSORT.LOST_TRACK_TTL,
            )
        )
    outputs = [None] * nr_sources
    display_outputs = [None] * nr_sources

    event_detectors = [None] * nr_sources
    event_logger = EventLogger(opt.event_log)
    if not opt.no_events:
        event_cfg = get_config()
        event_cfg.merge_from_file(opt.event_config)
        if event_cfg.get('EVENTS', {}).get('ENABLED', True):
            event_detectors = [load_event_config(event_cfg) for _ in range(nr_sources)]
    watchlist = WatchlistDB(
        opt.watchlist_db,
        body_threshold=opt.watch_body_threshold,
        face_threshold=opt.watch_face_threshold,
        min_intrusions=opt.watch_min_intrusions,
        min_loitering=opt.watch_min_loitering,
    ) if opt.watchlist_db else None

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference. MOT's official detector boxes can be used to avoid
        # YOLO misses in dense crowds; they are already in original-image coordinates.
        if mot_detections is not None:
            frame_detections = mot_detections.get(frame_idx + 1, [])
            pred = [torch.tensor(frame_detections, device=device)]
            t3 = time_sync()
        else:
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
            pred = model(im, augment=opt.augment, visualize=visualize)
            t3 = time_sync()
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[1] += t3 - t2
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            if opt.mot_sequence:
                txt_file_name = opt.mot_sequence
            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            current_events = []
            current_statuses = {}

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                if mot_detections is None:
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                if len(outputs[i]) > 0:
                    for output in outputs[i]:
                        track_id = output[4]
                        if save_txt:
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # MOTChallenge result format:
                            # frame, id, left, top, width, height, confidence, x, y, z
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (
                                    frame_idx + 1, track_id, bbox_left, bbox_top,
                                    bbox_w, bbox_h, -1, -1, -1, -1
                                ))

                        if save_crop:
                            c = int(output[5])
                            crop_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                            save_one_box(
                                output[0:4], imc,
                                file=save_dir / 'crops' / crop_name / names[c] / f'{track_id}' / f'{p.stem}.jpg',
                                BGR=True,
                            )

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                outputs[i] = np.empty((0, 6), dtype=int)
                deepsort_list[i].increment_ages()
                LOGGER.info('No detections')

            if event_detectors[i] is not None:
                if vid_cap is not None:
                    fps = float(vid_cap.get(cv2.CAP_PROP_FPS) or 0.0)
                elif webcam:
                    fps = 30.0
                else:
                    fps = opt.fps or 15.0
                current_events, current_statuses = event_detectors[i].update(
                    outputs[i], frame_idx + 1, fps=fps
                )

            watchlist_statuses = {}
            if watchlist is not None:
                profiles = {
                    int(output[4]): deepsort_list[i].get_track_profile(int(output[4]))
                    for output in outputs[i]
                }
                watchlist_statuses = watchlist.observe(
                    profiles,
                    current_events,
                    manual_track_id=opt.watch_track_id,
                    manual_label=opt.watch_label,
                )
            for event in current_events:
                target = watchlist_statuses.get(int(event['track_id']))
                if target is not None:
                    event['watchlist_id'] = target['target_id']
                LOGGER.warning(
                    'EVENT %s: ID %s%s', event['event'], event['track_id'],
                    f" in {event['zone']}" if event['zone'] else ''
                )
                event_logger.write([event])

            display_outputs[i] = deepsort_list[i].get_predicted_outputs(
                opt.display_track_age
            )
            display_watchlist_statuses = dict(watchlist_statuses)
            if watchlist is not None:
                for output in display_outputs[i]:
                    track_id = int(output[4])
                    if track_id not in display_watchlist_statuses:
                        target = watchlist.status_for_track(track_id)
                        if target is not None:
                            display_watchlist_statuses[track_id] = target

            # Stream results
            im0 = annotator.result()
            if event_detectors[i] is not None:
                im0 = event_detectors[i].draw(
                    im0, current_statuses, current_events, frame_idx + 1,
                    display_watchlist_statuses if watchlist is not None else None,
                )
            if save_vid or show_vid:
                for output in display_outputs[i]:
                    bboxes = output[0:4]
                    track_id = int(output[4])
                    class_id = int(output[5])
                    target = (display_watchlist_statuses or {}).get(track_id)
                    color = (0, 0, 255) if target is not None else (0, 255, 0)
                    label = (
                        f"{target['label']} / ID {track_id}"
                        if target is not None else f"ID {track_id} {names[class_id]}"
                    )
                    annotator.box_label(bboxes, label, color=color)
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_ibn_x1_0_MSMT17')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--event-config', type=str, default='deep_sort/configs/events.yaml',
                        help='restricted-area and loitering event configuration')
    parser.add_argument('--event-log', type=str, default='',
                        help='optional JSONL file for intrusion/loitering events')
    parser.add_argument('--no-events', action='store_true', help='disable event detection and drawing')
    parser.add_argument('--fps', type=float, default=0.0,
                        help='FPS override; 0 reads MOT seqinfo.ini or uses 15 for image folders')
    parser.add_argument('--mot-root', type=str, default='',
                        help='MOT17/MOT20 root; enables standard sequence input')
    parser.add_argument('--mot-dataset', choices=['MOT17', 'MOT20'], default='MOT17')
    parser.add_argument('--mot-split', choices=['train', 'test'], default='train')
    parser.add_argument('--mot-sequence', type=str, default='',
                        help='MOT sequence name, for example MOT17-04-SDP')
    parser.add_argument('--mot-use-det', action='store_true',
                        help='use the sequence det/det.txt boxes instead of YOLO detections')
    parser.add_argument('--mot-det-conf', type=float, default=0.2,
                        help='minimum confidence for MOT det.txt boxes')
    parser.add_argument('--ucf-root', type=str, default='',
                        help='extracted UCF-Crime root; enables UCF video selection')
    parser.add_argument('--ucf-category', type=str, default='',
                        help='UCF-Crime category, for example Burglary or Stealing')
    parser.add_argument('--ucf-video', type=str, default='',
                        help='one UCF-Crime video name or relative path')
    parser.add_argument('--ucf-split', choices=['all', 'train', 'test'], default='all',
                        help='official UCF-Crime anomaly detection split to validate')
    parser.add_argument('--watchlist-db', type=str, default='runs/watchlist.sqlite3',
                        help='SQLite database for persistent watchlist targets; empty disables it')
    parser.add_argument('--watch-track-id', type=int, default=None,
                        help='manually lock this current tracker ID into the watchlist')
    parser.add_argument('--watch-label', type=str, default='suspect',
                        help='label assigned to a manually locked target')
    parser.add_argument('--watch-min-intrusions', type=int, default=1,
                        help='zone-entry events required alongside loitering for automatic promotion')
    parser.add_argument('--watch-min-loitering', type=int, default=1,
                        help='loitering events required for automatic watchlist promotion')
    parser.add_argument('--watch-body-threshold', type=float, default=0.35,
                        help='cosine distance threshold for body ReID watchlist matching')
    parser.add_argument('--watch-face-threshold', type=float, default=0.40,
                        help='cosine distance threshold for face ReID watchlist matching')
    parser.add_argument('--display-track-age', type=int, default=15,
                        help='keep confirmed predicted boxes visible for this many missed frames')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    if opt.mot_root and opt.ucf_root:
        parser.error('--mot-root and --ucf-root cannot be used together')
    if opt.ucf_root:
        if opt.mot_use_det:
            parser.error('--mot-use-det is only valid with --mot-root')
        try:
            opt.source = resolve_ucf_source(
                opt.ucf_root,
                category=opt.ucf_category,
                video=opt.ucf_video,
                split=opt.ucf_split,
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
    elif opt.ucf_category or opt.ucf_video or opt.ucf_split != 'all':
        parser.error('--ucf-category, --ucf-video and --ucf-split require --ucf-root')

    if opt.mot_root:
        if not opt.mot_sequence:
            parser.error('--mot-sequence is required when --mot-root is used')
        mot_image_dir = resolve_mot_image_dir(
            opt.mot_root, opt.mot_dataset, opt.mot_split, opt.mot_sequence
        )
        opt.source = str(mot_image_dir)
        if not opt.fps:
            opt.fps = read_mot_fps(mot_image_dir) or 15.0
        opt.save_txt = True
    elif opt.mot_use_det:
        parser.error('--mot-use-det requires --mot-root and --mot-sequence')

    # This application tracks people. Apply the filter to ordinary videos as
    # well as MOT inputs so unrelated COCO classes (for example TV) can never
    # enter DeepSORT or appear in the output.
    if opt.classes is None:
        opt.classes = [0]

    with torch.no_grad():
        detect(opt)
