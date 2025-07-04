import argparse
import cv2
import os
import csv
from datetime import datetime
import sys
import platform
import numpy as np
from pathlib import Path
import torch
import requests
import threading
import queue

# Limitar uso de hilos por librerías
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# URL destino para enviar los datos agrupados
URL = "https://f61b-45-162-74-17.ngrok-free.app/detectar"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from ultralytics.yolo.data.utils import VID_FORMATS
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes
from ultralytics.yolo.utils.plotting import Annotator, colors

# Cola para pasar detecciones entre hilos
deteccion_queue = queue.Queue()
ultimo_segundo_enviado = None
lock = threading.Lock()

@torch.no_grad()
def run(
    source='voli.mp4',
    yolo_weights=WEIGHTS / 'yolov8n.pt',
    imgsz=(768, 768),
    conf_thres=0.25,
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
    is_file = Path(source).suffix[1:] in VID_FORMATS
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)

    classes = [0]  # Detectar solo personas (clase 0)

    device = select_device(device)
    model = AutoBackend(yolo_weights, device=device, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)

    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))

    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]

        preds = model(im)
        detections_per_image = non_max_suppression(preds, conf_thres, iou_thres, classes=classes, max_det=max_det)

        for i, det in enumerate(detections_per_image):
            if webcam:
                p, im0 = path[i], im0s[i].copy()
            else:
                p, im0 = path, im0s.copy()
            p = Path(p)

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            lista_detecciones = []

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                    lista_detecciones.append({'time': current_time, 'x': float(x_center), 'y': float(y_center)})

                    if not hide_labels:
                        label = f"{names[int(cls)]} {conf:.2f}" if not hide_class else f"{conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(cls, True))

            if lista_detecciones:
                deteccion_queue.put(lista_detecciones)

            im0 = annotator.result()
            if show_vid:
                if platform.system() == 'Linux':
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):
                    exit()


def procesar_detecciones_por_segundo(detecciones):
    global ultimo_segundo_enviado, lock

    if not detecciones:
        print("No hay detecciones para procesar")
        return
    try:
        tiempo_actual = datetime.strptime(detecciones[0]['time'], "%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        print("Error al parsear tiempo:", e)
        return

    segundo_actual = tiempo_actual.replace(microsecond=0)

    with lock:
        if ultimo_segundo_enviado != segundo_actual:
            ultimo_segundo_enviado = segundo_actual
            # Enviar TODAS las detecciones del frame de ese segundo
            enviar_payload(detecciones, segundo_actual)
        else:
            pass

def enviar_payload(detecciones, segundo):
    try:
        payload = {
            "timestamp": segundo.isoformat(),
            "personas": len(detecciones),
            "coordenadas": [{"x": d["x"], "y": d["y"]} for d in detecciones]
        }
        response = requests.post(URL, json=payload)
        print(f" Enviado JSON: {payload} | Respuesta: {response.status_code}")
    except Exception as e:
        print(f"❌ Error al enviar payload: {e}")

def hilo_envio_y_csv():
    global ultimo_segundo_enviado
    with open('detecciones.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Escribir cabecera solo si el archivo está vacío
        if csvfile.tell() == 0:
            writer.writerow(['timestamp', 'personas', 'coordenadas'])
        while True:
            lista_detecciones = deteccion_queue.get()
            if not lista_detecciones:
                continue
            try:
                tiempo_actual = datetime.strptime(lista_detecciones[0]['time'], "%Y-%m-%d %H:%M:%S.%f")
            except Exception as e:
                print("Error al parsear tiempo:", e)
                continue
            segundo_actual = tiempo_actual.replace(microsecond=0)
            with lock:
                if ultimo_segundo_enviado != segundo_actual:
                    ultimo_segundo_enviado = segundo_actual
                    enviar_payload(lista_detecciones, segundo_actual)
                    # Guardar en CSV
                    writer.writerow([
                        segundo_actual.isoformat(),
                        len(lista_detecciones),
                        str([{"x": d["x"], "y": d["y"]} for d in lista_detecciones])
                    ])
                    csvfile.flush()

if __name__ == '__main__':
    threading.Thread(target=hilo_envio_y_csv, daemon=True).start()

    run()