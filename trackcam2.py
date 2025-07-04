import argparse
import cv2
import os
import csv
from datetime import datetime

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"


import sys
import platform
import numpy as np
from pathlib import Path
import torch
import requests
import threading

URL = "https://2300-45-162-74-17.ngrok-free.app/detectar"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from ultralytics.yolo.data.utils import VID_FORMATS
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes
from ultralytics.yolo.utils.plotting import Annotator, colors

from trackers.multi_tracker_zoo import create_tracker
from sender.jsonlogger import JsonLogger
from sender.szmq import ZmqLogger

@torch.no_grad()
def run(
        source='sc2.mp4',
        yolo_weights=WEIGHTS / 'yolov8n.pt',
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',
        tracking_method='strongsort',
        tracking_config='trackers/strongsort/configs/strongsort.yaml',
        imgsz=(640, 640),
        conf_thres=0.4,
        iou_thres=0.45,
        max_det=100,
        device='0',
        show_vid=True,
        half=False,
        vid_stride=1,
        line_thickness=2,
        hide_labels=False,
        hide_conf=False,
        hide_class=False,
       
):

    source = str(source)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download if needed

    # Solo la clase 0 (persona)
    classes = [0]

    # Load model
    device = select_device(device)
    model = AutoBackend(yolo_weights, device=device, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)


    csv_file = open(f"{Path(source).stem}_tracking.csv", mode="w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["ncam", "id", "time", "x", "y"])
    csv_writer.writeheader()

    # Data loader
    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            vid_stride=vid_stride
        )

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))

    # Crear trackers para cada fuente
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker)

    outputs = [None] * bs
    js_logger = JsonLogger(f"{source}-{tracking_method}.log")

    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        with torch.profiler.profile() as dt:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]

            preds = model(im)

            p = non_max_suppression(preds, conf_thres, iou_thres, classes=classes, max_det=max_det)

        for i, det in enumerate(p):
            if webcam:
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)

            curr_frames[i] = im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                outputs[i] = tracker_list[i].update(det.cpu(), im0)

                if len(outputs[i]) > 0:
                    for output in outputs[i]:
                        bbox = output[0:4].tolist()
                        id = int(output[4])
                        cls = int(output[5])
                        conf = float(output[6])

                        x_center = (bbox[0] + bbox[2]) / 2
                        y_center = (bbox[1] + bbox[3]) / 2
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                        log2 = {'ncam': 2, 'id': id, 'time': current_time, 'x': float(x_center), 'y': float(y_center)}
                        threading.Thread(target=send_data, args=(log2,)).start()
                        threading.Thread(target=save_to_csv, args=(log2,)).start()

                        # Etiquetas para mostrar
                        label = None if hide_labels else (
                            f'{id} {names[cls]}' if hide_conf else
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {names[cls]} {conf:.2f}')
                        )
                        color = colors(cls, True)
                        annotator.box_label(bbox, label, color=color)

            im0 = annotator.result()
            if show_vid:
                if platform.system() == 'Linux':
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):
                    exit()

            prev_frames[i] = curr_frames[i]

    csv_file.close()

def save_to_csv(log_data, filename="cam2_tracking.csv"):
    file_exists = os.path.isfile(filename)
    try:
        with open(filename, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ncam", "id", "time", "x", "y"])
            if not file_exists or os.stat(filename).st_size == 0:
                writer.writeheader()
            writer.writerow(log_data)
    except Exception as e:
        print("Error al guardar en CSV:", e)

def send_data(log_data):
    try:
        response = requests.post(URL, json=log_data)
        print(f"✅ Enviado: {log_data} | Respuesta: {response.status_code}")
    except Exception as e:
        print("Error al enviar:", e)

if __name__ == '__main__':
    run()
