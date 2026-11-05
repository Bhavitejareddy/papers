import os, sys, torch

torch_dir = os.path.dirname(torch.__file__)
torch_lib = os.path.join(torch_dir, "lib")
os.add_dll_directory(torch_lib)
backend1 = os.path.join(os.path.dirname(__file__), "fused_backend")
os.add_dll_directory(backend1)
sys.path.insert(0, backend1)
backend2 = os.path.join(os.path.dirname(__file__), "build_roi")
os.add_dll_directory(backend2)
sys.path.insert(1, backend2)
import roi_fused
import fused_preprocess_new  # kernel for 560*560 image
import fused_preprocess_large  # kernel for 1120*1120
import split_concat
import fire_smoke_preprocess
import pano_resize_encoding_optimize_new

backend3 = os.path.join(os.path.dirname(__file__), "scatter_labels")
os.add_dll_directory(backend3)
sys.path.insert(2, backend3)
import scatter_fused
import subprocess
from ultralytics import YOLO
from fire_smoke_final_latest.infer import video_analytics_predict_dual_yolov26

from collections import defaultdict
from threading import Lock
from queue import Empty
import av
import av.error
import PyNvVideoCodec as nvc

# NVDEC packet wrapper: retain the backing NumPy buffer until Decode() returns.
_PacketData = nvc._PyNvVideoCodec.PacketData

def make_packet(buf):
    arr = np.frombuffer(buf, np.uint8)
    packet = _PacketData()
    packet.bsl_data = arr.ctypes.data
    packet.bsl = arr.nbytes
    return packet, arr

def sync_current_stream():
    event = torch.cuda.Event()
    event.record()
    event.synchronize()
from eventlet.queue import Full
# import kornia.geometry.transform as KT
# import kornia

import logging

logger = logging.getLogger(__name__)

ANALYTICS_RESULT = {}
ANALYTICS_LOCK = Lock()

import eventlet

eventlet.monkey_patch()
from config import settings as s
from src.utils.preProcessing import get_classes
from src.blocks.detection import detectionModel, detectionModelRFDETRLarge
from src.blocks.classification import classificationModel
import torch
import threading
import time
import numpy as np
import cv2
import torch
import av
from queue import Full, Empty
import torch.nn.functional as F
from config.sharedConfig import sharedData as sh

from turbojpeg import TurboJPEG, TJFLAG_FASTUPSAMPLE
from queue import Queue
import base64

PIPELINE_QUEUE_SIZE = 4
import socketio
# os.makedirs(OUTPUT_DIR, exist_ok=True)

from concurrent.futures import ThreadPoolExecutor

encode_pool = ThreadPoolExecutor(max_workers=2)


def gpu_resize(tensor, out_w=2000, out_h=270):
    # (H, W, 3) → (1, 3, H, W) float
    t = tensor.permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(out_h, out_w),
                      mode='bilinear', align_corners=False)
    # (1, 3, H, W) → (H, W, 3) uint8
    return t.squeeze(0).permute(1, 2, 0).byte()


def gpu_resize_kornia(tensor, out_w=2000, out_h=270):
    # tensor: (H, W, 3) uint8
    t = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = KT.resize(t, (out_h, out_w), antialias=False)
    return (t.squeeze(0).permute(1, 2, 0) * 255).byte()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_MAP_FIRE_SMOKE = {"fire": 0, "smoke": 1}
CLASS_MAP_HUMAN_CONFLICT = {"violence": 2}
CLASS_MAP_FIRE_SMOKE_model = {0: "fire", 1: "smoke"}
# CLASS_MAP_FIRE = {"fire":0, "smoke":1}

CLASS_MAP_HUMAN_CONFLICT_model = {0: "non_violence", 1: "violence"}
# YOLO_MODEL_FIRE_SMOKE= "E:/aice_ins_releaseversion/fire_smoke_final_latest/pre_trained_weights/best.pt"
# YOLO_MODEL_HUMAN_CONFLICT= "E:/aice_ins_releaseversion/violence_detection_yolov8\Yolo_nano_weights.pt"
# YOLO_MODEL_FIRE= torch.hub.load('ultralytics/yolov5', 'custom', path='fire/yolov5s_best.pt')
# YOLO_MODEL_SMOKE= torch.hub.load('ultralytics/yolov5', 'custom', path='fire/smoke.pt')

# YOLO_MODEL_HUMAN_CONFLICT= "./violence_detection_yolov8\Yolo_nano_weights.pt"
# input_images = ["./output2_8000.jpg", "./output_back_8000.jpg"]
CONFIDENCE_THRESHOLD_FIRE = 0.1
CONFIDENCE_THRESHOLD_SMOKE = 0.4

model_yolo_fire = YOLO(r".\fire\YOLOv26_Fire-Smoke.pt").to(device=DEVICE)
model_yolo_smoke = YOLO(r".\fire\YOLOv26_Fire-Smoke.pt").to(device=DEVICE)

import torch
import torch.nn.functional as F


@torch.no_grad()
def resize_with_letterbox(crop, final_size=384, pad_value=128):
    """
    crop: [C, H, W]
    returns: [C, final_size, final_size]

    Preserves aspect ratio and uses letterbox padding.
    """

    C, H, W = crop.shape

    # -----------------------------------------
    # Handle empty crops
    # -----------------------------------------
    if H == 0 or W == 0:
        return torch.full(
            (C, final_size, final_size),
            pad_value,
            device=crop.device,
            dtype=torch.float32
        )

    # -----------------------------------------
    # IMPORTANT:
    # uint8 -> float32 before interpolation
    # -----------------------------------------
    crop = crop.float()

    # -----------------------------------------
    # Calculate scale preserving aspect ratio
    # -----------------------------------------
    scale = min(
        final_size / W,
        final_size / H
    )

    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))

    # -----------------------------------------
    # Resize
    # [C,H,W] -> [1,C,H,W]
    # -----------------------------------------
    resized = F.interpolate(
        crop.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    # -----------------------------------------
    # Calculate letterbox padding
    # -----------------------------------------
    pad_w = final_size - new_w
    pad_h = final_size - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # -----------------------------------------
    # Letterbox to final_size × final_size
    # -----------------------------------------
    output = F.pad(
        resized,
        (
            pad_left,
            pad_right,
            pad_top,
            pad_bottom
        ),
        mode="constant",
        value=float(pad_value)
    )

    return output


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import human_violence_model.x3d_net as x3d


class HumanConflictDetector:

    def __init__(
            self,
            model_path,
            device="cuda:0",
            clip_len=16,
            input_size=224,
            threshold=0.80
    ):
        self.device = torch.device(device)

        self.clip_len = clip_len
        self.input_size = input_size
        self.threshold = threshold

        print("[HUMAN CONFLICT] Loading X3D model...")

        self.model = x3d.generate_model(
            x3d_version='M',
            n_classes=2,
            n_input_channels=3,
            task='class',
            dropout=0,
            base_bn_splits=1
        )

        state_dict = torch.load(
            model_path,
            map_location='cpu'
        )

        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.model.half()

        self.softmax = nn.Softmax(dim=1)

        self.mean = torch.tensor(
            [0.45, 0.45, 0.45],
            device=self.device,
            dtype=torch.float16
        ).view(1, 3, 1, 1, 1)

        self.std = torch.tensor(
            [0.225, 0.225, 0.225],
            device=self.device,
            dtype=torch.float16
        ).view(1, 3, 1, 1, 1)

        print("[HUMAN CONFLICT] Model loaded")

    @torch.inference_mode()
    def predict(self, clip):
        """
        clip: list of BGR frames
        """

        np_clip = np.stack(clip)

        inputs = torch.from_numpy(np_clip)

        inputs = inputs.pin_memory()

        inputs = inputs.to(
            self.device,
            non_blocking=True
        )

        # [T,H,W,C] → [1,C,T,H,W]
        inputs = inputs.permute(
            3, 0, 1, 2
        ).unsqueeze(0)

        inputs = inputs.half()

        # Resize
        inputs = inputs.squeeze(0)

        inputs = inputs.permute(
            1, 0, 2, 3
        )

        inputs = F.interpolate(
            inputs,
            size=(self.input_size,
                  self.input_size),
            mode='bilinear',
            align_corners=False
        )

        inputs = inputs.permute(
            1, 0, 2, 3
        ).unsqueeze(0)

        inputs = inputs.div_(255.0)

        inputs = inputs.sub_(self.mean)

        inputs = inputs.div_(self.std)

        outputs = self.model(inputs)

        if outputs.ndim == 3:
            outputs = outputs.squeeze(2)

        probs = self.softmax(outputs)

        fight_prob = probs[0, 0].item()

        pred = outputs.argmax(1).item()

        is_fight = (
                pred == 0 and
                fight_prob >= self.threshold
        )

        return is_fight, fight_prob


human_conflict_detector = HumanConflictDetector(
    model_path="./human_violence_model/pre_trained_x3d_model.pt",
    device="cuda:0",
    clip_len=16,
    threshold=0.80
)

NUM_ZONES = 2
ZONE_W = 3840

FRONT_CONFLICT_BUFFERS = [
    deque(maxlen=human_conflict_detector.clip_len)
    for _ in range(NUM_ZONES)
]

REAR_CONFLICT_BUFFERS = [
    deque(maxlen=human_conflict_detector.clip_len)
    for _ in range(NUM_ZONES)
]

# FRONT_CONFLICT_BUFFER = deque(
#     maxlen=human_conflict_detector.clip_len
# )
#
# REAR_CONFLICT_BUFFER = deque(
#     maxlen=human_conflict_detector.clip_len
# )


# CLASS_NAMES = {
#     0: "Background",
#     1: "SeaVessel",
#     2: "Cargo",
#     3: "Fishing",
#     4: "Boat"
# }
# Create a Socket.IO server
sio = socketio.Server(async_mode="eventlet", cors_allowed_origins="*")
app = socketio.WSGIApp(sio)

queue_1 = Queue(maxsize=1)
queue_2 = Queue(maxsize=1)
analytics_front = Queue(maxsize=1)
analytics_back = Queue(maxsize=1)
emit_queue = Queue()
analytics_queue_fire_smoke = Queue(maxsize=2)
analytics_queue_human_conflict = Queue(maxsize=2)

tiles_data = [
    {'panorama_idx': 0, 'x_offset': 0},
    {'panorama_idx': 0, 'x_offset': 1120},
    {'panorama_idx': 0, 'x_offset': 2240},
    {'panorama_idx': 0, 'x_offset': 3360},
    {'panorama_idx': 0, 'x_offset': 4480},
    {'panorama_idx': 0, 'x_offset': 5600},
    {'panorama_idx': 0, 'x_offset': 6720},
    {'panorama_idx': 1, 'x_offset': 0},
    {'panorama_idx': 1, 'x_offset': 1120},
    {'panorama_idx': 1, 'x_offset': 2240},
    {'panorama_idx': 1, 'x_offset': 3360},
    {'panorama_idx': 1, 'x_offset': 4480},
    {'panorama_idx': 1, 'x_offset': 5600},
    {'panorama_idx': 1, 'x_offset': 6720},
]

# ================= OPTIONAL PIPELINE TOGGLES =================
DETECTION_STATE = {"/front": False, "/rear": False}
CLASSIFICATION_STATE = {"/front": False, "/rear": False}

EVENT_STATE = {
    "/front": {
        "fire": False,
        "smoke": False,
        "human_conflict": False
    },
    "/rear": {
        "fire": False,
        "smoke": False,
        "human_conflict": False
    }
}


@sio.event(namespace="/front")
def connect(sid, environ):
    print("✅ Front namespace connected:", sid)


@sio.event(namespace="/rear")
def connect(sid, environ):
    print("✅ Rear namespace connected:", sid)


@sio.on("set_event", namespace="/front")
def set_event_front(sid, data):
    etype = data.get("type")
    # print("etype",etype)
    enabled = bool(data.get("enabled", False))
    if etype in EVENT_STATE["/front"]:
        EVENT_STATE["/front"][etype] = enabled
        print(f"🔁 /front event {etype} = {enabled}")


@sio.on("set_event", namespace="/rear")
def set_event_rear(sid, data):
    etype = data.get("type")
    enabled = bool(data.get("enabled", False))
    if etype in EVENT_STATE["/rear"]:
        EVENT_STATE["/rear"][etype] = enabled
        print(f"🔁 /rear event {etype} = {enabled}")


@sio.on("set_camera_mode", namespace="/front")
def set_camera_mode(sid, data):
    mode = int(data.get("mode", 0))
    if mode in [0, 1]:
        sh.CAMERA_MODE = mode
        print(f"Camera mode /front = {mode}")
    else:
        print(f"Invalid camera mode received: {mode}")


@sio.on("set_classification", namespace="/front")
def set_classification_front(sid, data):
    CLASSIFICATION_STATE["/front"] = bool(data.get("enabled", False))
    print("Classification /front =", CLASSIFICATION_STATE["/front"])


@sio.on("set_classification", namespace="/rear")
def set_classification_rear(sid, data):
    CLASSIFICATION_STATE["/rear"] = bool(data.get("enabled", False))
    print("Classification /rear =", CLASSIFICATION_STATE["/rear"])


@sio.on("set_detection", namespace="/front")
def set_detection_front(sid, data):
    DETECTION_STATE["/front"] = bool(data.get("enabled", False))
    print("Detection /front =", DETECTION_STATE["/front"])


@sio.on("set_detection", namespace="/rear")
def set_detection_rear(sid, data):
    DETECTION_STATE["/rear"] = bool(data.get("enabled", False))
    print("Detection /rear =", DETECTION_STATE["/rear"])


def start_server():
    sio.start_background_task(emit_worker)
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5000)), app)


# detectionModelCCD = detectionModel(CCD_DETECTION="C:/Users/CRL-AI2/Desktop/super_computer_resources/weight_files/seaVessel_detection_ep_1000_RF_DETR_rs_560/checkpoint_best_total.pth"
# ,DET_THRES=0.2)
detectionModelCCD = detectionModelRFDETRLarge(
    DETECTION_MODEL="models\ccdDetection\seaVessel_detection_CCD_RF_DETRLarge_rs_1120_updated_stats_b16/checkpoint_best_ema.pth"
    , DET_THRES=0.2)
_, classes2int_dict = get_classes(s.CLASSES_UI)
detectionModelIR = detectionModel(
    CCD_DETECTION=r".\models\irDetection\seaVessel_detection_ir_ep_1000_RF_DETR_rs_560\checkpoint_best_total.pth"
    , DET_THRES=0.1)
classificationModelCCD = classificationModel(s.CCD_CLASSIFIER_MODEL_PATH, s.CCD_CLASSIFICATION,
                                             s.CLASSIFICATION_CLASSES_CCD, classes2int_dict, s.CLS_THRES_CCD,
                                             s.CLS_ACC_LIMIT_CCD)
classificationModelIR = classificationModel(s.IR_CLASSIFIER_MODEL_PATH, s.IR_CLASSIFICATION,
                                            s.CLASSIFICATION_CLASSES_IR, classes2int_dict, s.CLS_THRES_IR,
                                            s.CLS_ACC_LIMIT_IR)

# # model_yolo_fire_smoke = YOLO(model=YOLO_MODEL_FIRE_SMOKE).to(device=DEVICE)
# model_yolo_fire = YOLO_MODEL_FIRE.to(DEVICE)
# model_yolo_smoke = YOLO_MODEL_SMOKE.to(DEVICE)
# model_yolo_human_conflict = YOLO(model=YOLO_MODEL_HUMAN_CONFLICT).to(device=DEVICE)


# offline input images
jpeg_turbo = TurboJPEG(lib_path="C:/windows/system32/turbojpeg.dll")


def encode_frame(frame):
    buf = jpeg_turbo.encode(frame, quality=70)
    return base64.b64encode(buf).decode()


def create_capture(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)

    # Reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open RTSP stream: {rtsp_url}")

    return cap


def emit_worker():
    while True:
        event_name, data, frame_id = emit_queue.get()
        print("into_emit", time.time())
        if event_name is None:
            break
        try:
            print(f"Emitting {event_name} inside Eventlet")
            print("event_name", event_name)
            if data:
                print("yesssss")
            sio.emit(event_name, data)
            # print(f"emitted{frame_id}",time.time())
            # eventlet.sleep(0.01)
        except Exception as e:
            print("Emit error:", e)
        eventlet.sleep(0)


# SOURCE_MODE   = 0        # 1 = RTSP live streams  |  0 = offline video files

# RTSP_URL_1    = "rtsp://your_cam1_url"
# RTSP_URL_2    = "rtsp://your_cam2_url"

# # VIDEO_PATH_1  = "E:/videos/vlc-record-2026-04-03-17h12m25s-wolfpack_stream1_cut.mp4-.mp4"
# # VIDEO_PATH_2  = "E:/videos/vlc-record-2026-04-03-17h12m25s-wolfpack_stream1_cut.mp4-.mp4"

# VIDEO_PATH_1  = "E:/videos/Detection_1.mp4"
# VIDEO_PATH_2  = "E:/videos/Detection_2.mp4"


SOURCE_MODE = 1  # 1 = RTSP live streams  |  0 = offline video files

RTSP_URL_DAY_FRONT = "rtsp://192.168.100.10:554/rtpstream/config1"
RTSP_URL_DAY_REAR = "rtsp://192.168.100.10:554/rtpstream/config1"

RTSP_URL_IR_FRONT = "rtsp://192.168.100.10:554/rtpstream/config1"
RTSP_URL_IR_REAR = "rtsp://192.168.100.10:554/rtpstream/config1"

# # day detection latest
# VIDEO_PATH_1  = r"D:\Harbour Watch\harbour_watch_v1\harbour_watchBackend\Detection_2.mp4"
# VIDEO_PATH_2  = r"D:\Harbour Watch\harbour_watch_v1\harbour_watchBackend\Detection_1.mp4"
#
# # # ir for the detcetion latest
# VIDEO_PATH_IR1 = r"D:\Harbour Watch\harbour_watch_v1\harbour_watchBackend\Detection_2.mp4"
# VIDEO_PATH_IR2 = r"D:\Harbour Watch\harbour_watch_v1class RTSPReader:
#     """
#     NVDEC-based reader. One thread per stream, frames stay on the GPU.
#
#     get_latest()  -> (frame, timestamp, seq)
#         frame: torch.cuda.Tensor, (H, W, 3) uint8, RGB   (None until first frame)
#         timestamp: time.monotonic() at decode time
#         seq: increments on every new frame
#     """
#
#     STALL_TIMEOUT = 15.0
#     MAX_CONSECUTIVE_DECODE_ERRORS = 10
#
#     def __init__(self, url, name="cam", reconnect_delay=2.0,
#                  out_w=None, out_h=None,          # accepted for compatibility, ignored
#                  source_mode=1,
#                  video_path=None,
#                  gpu_id=0):
#         self.url = url
#         self.name = name
#         self.reconnect_delay = reconnect_delay
#         self.source_mode = source_mode
#         self.video_path = video_path
#         self.gpu_id = gpu_id
#         self.decode_frames = 0
#         self.decode_start_time = time.perf_counter()
#         self.decode_fps = 0.0
#
#         if self.source_mode == 0 and not self.video_path:
#             raise ValueError(f"[{self.name}] source_mode=0 but no video_path provided")
#
#         self._cond = threading.Condition()
#         self._latest = None
#         self._latest_ts = 0.0
#         self._seq = 0
#         self._new_frame_event = threading.Event()   # kept for compatibility
#         self._stop_event = threading.Event()
#
#         # ── Stats ──────────────────────────────────────────────────────────
#         self._stat_lock = threading.Lock()
#         self._reset_stats(time.time())
#         self._stat_report_interval = 10.0
#
#         self._thread = threading.Thread(
#             target=self._reader_loop, daemon=True, name=f"reader-{name}")
#         self._thread.start()
#
#     # ── stats helpers ─────────────────────────────────────────────────────
#
#     def _reset_stats(self, now):
#         self._stat_frames_decoded = 0
#         self._stat_decode_errors = 0
#         self._stat_demux_errors = 0
#         self._stat_packets_received = 0
#         self._stat_t_decode_total = 0.0   # time in Decode() (approximate, GPU is async)
#         self._stat_t_copy_total = 0.0     # time in GPU D2D clone
#         self._stat_t_lock_total = 0.0     # time acquiring the frame lock
#         self._stat_last_report = now
#
#     def _avg_ms(self, total):
#         n = self._stat_frames_decoded
#         return (total / n * 1000) if n else 0.0
#
#     def _maybe_report_stats(self):
#         now = time.time()
#         elapsed = now - self._stat_last_report
#         if elapsed < self._stat_report_interval:
#             return
#         print(
#             f"[{self.name}][stats] "
#             f"interval={elapsed:.1f}s | "
#             f"fps={self._stat_frames_decoded / elapsed:.1f} | "
#             f"decoded={self._stat_frames_decoded} | "
#             f"packets={self._stat_packets_received} | "
#             f"decode_errors={self._stat_decode_errors} "
#             f"demux_errors={self._stat_demux_errors} | "
#             f"avg_decode={self._avg_ms(self._stat_t_decode_total):.2f}ms "
#             f"avg_gpu_copy={self._avg_ms(self._stat_t_copy_total):.2f}ms "
#             f"avg_lock_wait={self._avg_ms(self._stat_t_lock_total):.2f}ms"
#         )
#         self._reset_stats(now)
#
#     def get_stats(self):
#         """Snapshot of current counters, callable from any thread."""
#         return {
#             "frames_decoded":   self._stat_frames_decoded,
#             "decode_errors":    self._stat_decode_errors,
#             "demux_errors":     self._stat_demux_errors,
#             "packets_received": self._stat_packets_received,
#             "avg_decode_ms":    self._avg_ms(self._stat_t_decode_total),
#             "avg_copy_ms":      self._avg_ms(self._stat_t_copy_total),
#             "avg_lock_ms":      self._avg_ms(self._stat_t_lock_total),
#         }
#
#     # ── open ──────────────────────────────────────────────────────────────
#
#     def _open(self):
#         src = self.url if self.source_mode == 1 else self.video_path
#         demux = nvc.CreateDemuxer(src)
#         dec = nvc.CreateDecoder(
#             gpuid=self.gpu_id,
#             codec=demux.GetNvCodecId(),
#             cudacontext=0,
#             cudastream=0,
#             usedevicememory=True,                       # frames stay on GPU
#             outputColorType=nvc.OutputColorType.RGB,    # colour conversion on GPU
#         )
#         try:
#             fps = float(demux.FrameRate()) or 25.0
#         except Exception:
#             fps = 25.0
#         try:
#             size = f"{demux.Width()}x{demux.Height()}"
#         except Exception:
#             size = "?"
#         print(f"[{self.name}] {'RTSP' if self.source_mode == 1 else 'file'} "
#               f"opened — {size} fps={fps:.2f}")
#         return demux, dec, fps
#
#     # ── decode loop (RTSP and file share it) ──────────────────────────────
#
#     def _run(self, demux, dec, fps):
#         is_file = self.source_mode == 0
#         period = 1.0 / fps
#         next_t = time.perf_counter()
#         consecutive_failures = 0
#         last_packet_time = time.time()
#         it = iter(demux)
#
#         print(f"[{self.name}] entering decode loop")
#
#         while not self._stop_event.is_set():
#             if not is_file and time.time() - last_packet_time > self.STALL_TIMEOUT:
#                 print(f"[{self.name}] stall detected, reconnecting…")
#                 return True
#
#             try:
#                 packet = next(it)
#             except StopIteration:
#                 print(f"[{self.name}] {'file ended, looping…' if is_file else 'stream ended'}")
#                 return True
#             except Exception as e:
#                 self._stat_demux_errors += 1
#                 print(f"[{self.name}] demux error: {e}")
#                 return True
#
#             if packet is None:
#                 continue
#
#             last_packet_time = time.time()
#             self._stat_packets_received += 1
#
#             # ── timed: NVDEC decode ───────────────────────────────────────
#             t0 = time.perf_counter()
#             try:
#                 surfaces = dec.Decode(packet)
#             except Exception as e:
#                 self._stat_decode_errors += 1
#                 consecutive_failures += 1
#                 print(f"[{self.name}] decode error ({consecutive_failures}): {e}")
#                 if consecutive_failures >= self.MAX_CONSECUTIVE_DECODE_ERRORS:
#                     print(f"[{self.name}] too many decode errors, reconnecting…")
#                     return True
#                 continue
#             self._stat_t_decode_total += time.perf_counter() - t0
#
#             for surf in surfaces:
#                 consecutive_failures = 0
#
#                 # ── timed: zero-copy view + one device-to-device copy ─────
#                 # (clone so the decoder can safely reuse its own buffer)
#                 t1 = time.perf_counter()
#                 frame = torch.from_dlpack(surf).clone()
#                 self._stat_t_copy_total += time.perf_counter() - t1
#
#                 # ── timed: publish ────────────────────────────────────────
#                 t2 = time.perf_counter()
#                 with self._cond:
#                     self._stat_t_lock_total += time.perf_counter() - t2
#                     self._latest = frame
#                     self._latest_ts = time.monotonic()
#                     self._seq += 1
#                     self._cond.notify_all()
#                 self._new_frame_event.set()
#                 self._stat_frames_decoded += 1
#
#                 # pace file playback to its native frame rate (deadline-based)
#                 # if is_file:
#                 #     next_t += period
#                 #     delay = next_t - time.perf_counter()
#                 #     if delay > 0:
#                 #         time.sleep(delay)
#                 #     else:
#                 #         next_t = time.perf_counter()
#
#             self._maybe_report_stats()
#
#         return False
#
#     # ── thread entry ──────────────────────────────────────────────────────
#
#     def _reader_loop(self):
#         print(f"[{self.name}] reader thread started "
#               f"(mode={'RTSP' if self.source_mode == 1 else 'file'})")
#         torch.cuda.set_device(self.gpu_id)
#
#         while not self._stop_event.is_set():
#             demux = dec = None
#             should_reconnect = True
#             try:
#                 demux, dec, fps = self._open()
#                 should_reconnect = self._run(demux, dec, fps)
#             except Exception as e:
#                 print(f"[{self.name}] exception: {e}")
#                 should_reconnect = True
#             finally:
#                 del dec, demux
#
#             if not self._stop_event
def build_and_emit_front_back_payloads(
        refined_outputs,
        TILE_W,
        frame_id1,
        frame_in_bytes_front,
        frame_in_bytes_back,
        cpu_buf3, cpu_buf4
):
    list_of_targets_front = []
    list_of_targets_back = []

    orig_w = 7680
    orig_h = 1080

    new_w = 2000
    new_h = 270

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    print(scale_x, scale_y)

    if sh.CAMERA_MODE == 1:
        num_tile = 4
    if sh.CAMERA_MODE == 0:
        num_tile = 7

    for tile_idx, det in enumerate(refined_outputs):

        # ---------------------------------
        # FRONT TILES
        # ---------------------------------

        if tile_idx < num_tile:

            pano_type = "front"
            local_tile_idx = tile_idx

        # ---------------------------------
        # BACK TILES
        # ---------------------------------

        else:

            pano_type = "back"
            local_tile_idx = tile_idx - num_tile

        offset_x = local_tile_idx * TILE_W

        # ---------------------------------
        # PROCESS DETECTIONS
        # ---------------------------------
        # print("tile",tile_idx)

        for box, score, cls in zip(
                det.xyxy,
                det.confidence,
                det.class_id):
            # print("raw_bbox",box.tolist())

            x1, y1, x2, y2 = box.tolist()

            x1 += offset_x
            x2 += offset_x
            # Uncomment these 4 lines for bbox scaling
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            print("bboxes", x1, y1, x2, y2, score, int(cls))

            temp_dict = {
                "xmin": x1,
                "xmax": x2,
                "ymin": y1,
                "ymax": y2,
                "type_of_target": int(cls),
                "confidence": float(score)
            }

            if pano_type == "front":
                list_of_targets_front.append(temp_dict)

            else:
                list_of_targets_back.append(temp_dict)
            # print("list_of_targets_front",list_of_targets_front)
            # print("list_of_targets_back", list_of_targets_back)

    # list_of_targets_front = []
    # list_of_targets_back = []
    #
    # # OPTIONAL: detailed tile-level breakdown
    # offsets1 = [i * TILE_W for i in range(14)]
    #
    # for det, offset_x in zip(refined_outputs, offsets1):
    #     for box, score, cls in zip(det.xyxy, det.confidence, det.class_id):
    #
    #         x1, y1, x2, y2 = box.tolist()
    #
    #         x1 += offset_x
    #         x2 += offset_x
    #         type_of_target = int(cls)
    #
    #         temp_dict = {}
    #
    #         if offset_x >= 7680:
    #             x1 = x1 - 7680
    #             x2 = x2 - 7680
    #
    #             temp_dict["xmin"] = x1
    #             temp_dict["xmax"] = x2
    #             temp_dict["ymin"] = y1
    #             temp_dict["ymax"] = y2
    #             temp_dict["type_of_target"] = type_of_target
    #
    #             list_of_targets_back.append(temp_dict)
    #
    #         else:
    #             temp_dict["xmin"] = x1
    #             temp_dict["xmax"] = x2
    #             temp_dict["ymin"] = y1
    #             temp_dict["ymax"] = y2
    #             temp_dict["type_of_target"] = type_of_target
    #
    #             list_of_targets_front.append(temp_dict)

    # ---------------- FRONT ----------------

    # =========================================================
    # APPLY MORPHOLOGICAL FUSION
    # =========================================================

    # list_of_targets_front = morphological_bbox_fusion(frame_id1,
    #     list_of_targets_front,
    #     pano_w=7680,
    #     pano_h=1080,
    #     kernel_size=25,
    #     iterations=1,
    #     save_dir="./pano"
    # )
    #
    # list_of_targets_back = morphological_bbox_fusion(frame_id1,
    #     list_of_targets_back,
    #     pano_w=7680,
    #     pano_h=1080,
    #     kernel_size=25,
    #     iterations=1,
    #     save_dir="./pano"
    # )

    # save_combined_pano(
    #     cpu_buf3,
    #     cpu_buf4,
    #     list_of_targets_front,
    #     list_of_targets_back,
    #     frame_id1
    # )

    targets_front = []
    for i, b in enumerate(list_of_targets_front, start=1):
        targets_front.append({
            "target_no": i,
            "xmin": int(b["xmin"]),
            "ymin": int(b["ymin"]),
            "xmax": int(b["xmax"]),
            "ymax": int(b["ymax"]),
            "type_of_target": int(b["type_of_target"])
        })

    payload_front = {
        "message_id": frame_id1,
        "sensor_id": 1,
        "panoramic_view_id": 1,
        "frame_id": frame_id1,
        "aice_epoch": int(time.time() * 1_000_000),
        "no_of_targets": len(targets_front),
        "targets": targets_front,
        "image": frame_in_bytes_front
    }

    # ---------------- BACK ----------------

    targets_back = []
    for i, b in enumerate(list_of_targets_back, start=1):
        targets_back.append({
            "target_no": i,
            "xmin": int(b["xmin"]),
            "ymin": int(b["ymin"]),
            "xmax": int(b["xmax"]),
            "ymax": int(b["ymax"]),
            "type_of_target": int(b["type_of_target"])

        })

    payload_back = {
        "message_id": frame_id1,
        "sensor_id": 1,
        "panoramic_view_id": 2,
        "frame_id": frame_id1,
        "aice_epoch": int(time.time() * 1_000_000),
        "no_of_targets": len(targets_back),
        "targets": targets_back,
        "image": frame_in_bytes_back
    }
    return payload_front, payload_back


import cv2
import numpy as np

global count

# def morphological_bbox_fusion(frame_id,
#         detections,
#         pano_w=7680,
#         pano_h=1080,
#         kernel_size=25,
#         iterations=1,
#         save_prefix="./pano/front"
# ):
#     """
#     detections:
#         list of dicts:
#         xmin,ymin,xmax,ymax,type_of_target
#
#     returns:
#         fused detections
#     """
#
#     # =========================================================
#     # 1. CREATE EMPTY BINARY IMAGE
#     # =========================================================
#
#     binary_before = np.zeros((pano_h, pano_w), dtype=np.uint8)
#
#     # =========================================================
#     # 2. DRAW ALL BBOXES AS WHITE
#     # =========================================================
#
#     for det in detections:
#
#         x1 = int(det["xmin"])
#         y1 = int(det["ymin"])
#         x2 = int(det["xmax"])
#         y2 = int(det["ymax"])
#
#         cv2.rectangle(
#             binary_before,
#             (x1, y1),
#             (x2, y2),
#             255,
#             -1
#         )
#
#     # =========================================================
#     # 3. SAVE BEFORE IMAGE
#     # =========================================================
#
#     cv2.imwrite(f"{save_prefix}_before.png{frame_id}", binary_before)
#
#     # =========================================================
#     # 4. MORPHOLOGICAL OPERATION
#     # =========================================================
#
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#
#     binary_after = cv2.morphologyEx(
#         binary_before,
#         cv2.MORPH_CLOSE,
#         kernel,
#         iterations=iterations
#     )
#
#     # =========================================================
#     # 5. SAVE AFTER IMAGE
#     # =========================================================
#
#     cv2.imwrite(f"{save_prefix}_after.png{frame_id}", binary_after)
#
#     # =========================================================
#     # 6. CONNECTED COMPONENTS
#     # =========================================================
#
#     num_labels, labels, stats, centroids = \
#         cv2.connectedComponentsWithStats(binary_after)
#
#     # =========================================================
#     # 7. CREATE FINAL FUSED BBOXES
#     # =========================================================
#
#     fused_detections = []
#
#     for i in range(1, num_labels):
#
#         x = stats[i, cv2.CC_STAT_LEFT]
#         y = stats[i, cv2.CC_STAT_TOP]
#         w = stats[i, cv2.CC_STAT_WIDTH]
#         h = stats[i, cv2.CC_STAT_HEIGHT]
#
#         fused_box = {
#             "xmin": int(x),
#             "ymin": int(y),
#             "xmax": int(x + w),
#             "ymax": int(y + h),
#             "type_of_target": 0
#         }
#
#         fused_detections.append(fused_box)
#
#     return fused_detections


import os
import cv2
import numpy as np


def morphological_bbox_fusion(
        frame_id,
        detections,
        pano_w=7680,
        pano_h=1080,
        kernel_size=25,
        iterations=1,
        save_dir="./pano/front"
):
    """
    detections:
        list of dicts:
        xmin,ymin,xmax,ymax,type_of_target,confidence

    returns:
        fused detections
    """

    # =========================================================
    # 0. CREATE SAVE DIRECTORY
    # =========================================================

    os.makedirs(save_dir, exist_ok=True)

    # =========================================================
    # 1. CREATE EMPTY BINARY IMAGE
    # =========================================================

    binary_before = np.zeros((pano_h, pano_w), dtype=np.uint8)

    # =========================================================
    # 2. DRAW ALL BBOXES AS WHITE
    # =========================================================

    for det in detections:
        x1 = int(det["xmin"])
        y1 = int(det["ymin"])
        x2 = int(det["xmax"])
        y2 = int(det["ymax"])

        cv2.rectangle(
            binary_before,
            (x1, y1),
            (x2, y2),
            255,
            -1
        )

    # =========================================================
    # 3. SAVE BEFORE IMAGE
    # =========================================================

    before_path = os.path.join(
        save_dir,
        f"{frame_id}_before.png"
    )

    cv2.imwrite(before_path, binary_before)

    # =========================================================
    # 4. MORPHOLOGICAL OPERATION
    # =========================================================

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    binary_after = cv2.morphologyEx(
        binary_before,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations
    )

    # =========================================================
    # 5. SAVE AFTER IMAGE
    # =========================================================

    after_path = os.path.join(
        save_dir,
        f"{frame_id}_after.png"
    )

    cv2.imwrite(after_path, binary_after)

    # =========================================================
    # 6. CONNECTED COMPONENTS
    # =========================================================

    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(binary_after)

    # =========================================================
    # 7. ASSIGN ORIGINAL DETECTIONS TO COMPONENTS
    # =========================================================

    component_to_detections = {}

    for det in detections:

        x1 = int(det["xmin"])
        y1 = int(det["ymin"])
        x2 = int(det["xmax"])
        y2 = int(det["ymax"])

        # center point of bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        component_id = labels[cy, cx]

        if component_id == 0:
            continue

        if component_id not in component_to_detections:
            component_to_detections[component_id] = []

        component_to_detections[component_id].append(det)

    # =========================================================
    # 8. CREATE FINAL FUSED BBOXES
    # =========================================================

    fused_detections = []

    for component_id, dets in component_to_detections.items():
        x = stats[component_id, cv2.CC_STAT_LEFT]
        y = stats[component_id, cv2.CC_STAT_TOP]
        w = stats[component_id, cv2.CC_STAT_WIDTH]
        h = stats[component_id, cv2.CC_STAT_HEIGHT]

        # -----------------------------------------------------
        # FIND CLASS WITH HIGHEST CONFIDENCE
        # -----------------------------------------------------

        best_det = max(
            dets,
            key=lambda d: d["confidence"]
        )

        final_class = best_det["type_of_target"]
        # final_conf = best_det["confidence"]

        fused_box = {
            "xmin": int(x),
            "ymin": int(y),
            "xmax": int(x + w),
            "ymax": int(y + h),
            "type_of_target": int(final_class)
        }

        fused_detections.append(fused_box)

    return fused_detections


def draw_bboxes(image, targets):
    vis = image.copy()

    for i, b in enumerate(targets, start=1):
        x1 = int(b["xmin"])
        y1 = int(b["ymin"])
        x2 = int(b["xmax"])
        y2 = int(b["ymax"])

        cls = int(b["type_of_target"])
        # conf = float(b.get("confidence", 0.0))

        # Map class ID to class name

        if 0 <= cls < len(CLASS_NAMES):
            class_name = CLASS_NAMES[cls]
        else:
            class_name = f"detect"

        # Draw bounding box
        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display class name and confidence
        label = f"{class_name}"

        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    return vis


def video_analytics_fire_smoke():
    print("[FIRE/SMOKE] thread started")

    while True:

        try:
            image_front, image_back, frame_id = analytics_queue_fire_smoke.get(timeout=0.1)
        except Empty:
            continue

        front_on_fire = EVENT_STATE["/front"]['fire']
        rear_on_fire = EVENT_STATE["/rear"]['fire']
        front_on_smoke = EVENT_STATE["/front"]['smoke']
        rear_on_smoke = EVENT_STATE["/rear"]['smoke']

        print("front_on_fire =", front_on_fire, "rear_on_fire =", rear_on_fire, "front_on_smoke =", front_on_smoke,
              "rear_on_smoke =", rear_on_smoke)

        if front_on_fire and rear_on_fire and front_on_smoke and rear_on_smoke:
            print("fire/smoke analytics running")
            print("image_for_fire", time.time())
            image_front = image_front[..., [2, 1, 0]]
            image_back = image_back[..., [2, 1, 0]]
            # offsets = [0, 1120, 2240, 3360, 4480, 5600, 6720]

            # tiles1 = fire_smoke_preprocess.run(image_front)
            # tiles2 = fire_smoke_preprocess.run(image_back)
            # import torch.nn.functional as F

            # HWC -> NCHW
            image_front = image_front.permute(2, 0, 1).unsqueeze(0)
            image_back = image_back.permute(2, 0, 1).unsqueeze(0)

            print(image_front.shape)

            # image_front = F.interpolate(
            #     image_front.bfloat16(),
            #     size=(640, 640),
            #     mode='bilinear',
            #     align_corners=False
            # )/ 255.0
            #
            # image_back = F.interpolate(
            #     image_back.bfloat16(),
            #     size=(640, 640),
            #     mode='bilinear',
            #     align_corners=False
            # )/ 255.0
            # all_tiles  = torch.cat([image_front, image_back], dim=0)

            # image_front : [1,3,1080,7680]
            # image_back  : [1,3,1080,7680]

            TILE_W = 1920
            NUM_TILES = 4

            # --------------------------------------------------
            # Split front panorama
            # [1,3,1080,7680] -> [4,3,1080,1920]
            # --------------------------------------------------
            front_tiles = (
                image_front
                    .view(1, 3, 1080, NUM_TILES, TILE_W)
                    .permute(3, 1, 2, 4, 0)
                    .squeeze(-1)
                    .contiguous()
            )

            # --------------------------------------------------
            # Split rear panorama
            # [1,3,1080,7680] -> [4,3,1080,1920]
            # --------------------------------------------------
            back_tiles = (
                image_back
                    .view(1, 3, 1080, NUM_TILES, TILE_W)
                    .permute(3, 1, 2, 4, 0)
                    .squeeze(-1)
                    .contiguous()
            )

            # --------------------------------------------------
            # Resize all 8 tiles together
            # --------------------------------------------------
            front_tiles = (
                    F.interpolate(
                        front_tiles.bfloat16(),
                        size=(640, 640),
                        mode="bilinear",
                        align_corners=False,
                    ) / 255.0
            )

            back_tiles = (
                    F.interpolate(
                        back_tiles.bfloat16(),
                        size=(640, 640),
                        mode="bilinear",
                        align_corners=False,
                    ) / 255.0
            )

            # --------------------------------------------------
            # Final batch
            # [8,3,640,640]
            # --------------------------------------------------
            all_tiles = torch.cat([front_tiles, back_tiles], dim=0)
            # all_tiles = torch.cat([tiles1, tiles2], dim=0)

            # tiles_data = [
            #     {'panorama_idx': 0, 'x_offset': 0},
            #     {'panorama_idx': 0, 'x_offset': 1120},
            #     {'panorama_idx': 0, 'x_offset': 2240},
            #     {'panorama_idx': 0, 'x_offset': 3360},
            #     {'panorama_idx': 0, 'x_offset': 4480},
            #     {'panorama_idx': 0, 'x_offset': 5600},
            #     {'panorama_idx': 0, 'x_offset': 6720},
            #     {'panorama_idx': 1, 'x_offset': 0},
            #     {'panorama_idx': 1, 'x_offset': 1120},
            #     {'panorama_idx': 1, 'x_offset': 2240},
            #     {'panorama_idx': 1, 'x_offset': 3360},
            #     {'panorama_idx': 1, 'x_offset': 4480},
            #     {'panorama_idx': 1, 'x_offset': 5600},
            #     {'panorama_idx': 1, 'x_offset': 6720},
            # ]
            # tiles_data = [
            #     {'panorama_idx': 0, 'x_offset': 0},
            #     {'panorama_idx': 1, 'x_offset':0}
            # ]

            tiles_data = [
                # Front panorama
                {'panorama_idx': 0, 'x_offset': 0},
                {'panorama_idx': 0, 'x_offset': 1920},
                {'panorama_idx': 0, 'x_offset': 3840},
                {'panorama_idx': 0, 'x_offset': 5760},

                # Rear panorama
                {'panorama_idx': 1, 'x_offset': 0},
                {'panorama_idx': 1, 'x_offset': 1920},
                {'panorama_idx': 1, 'x_offset': 3840},
                {'panorama_idx': 1, 'x_offset': 5760},
            ]

            boxes, scores, labels, pano_ids = video_analytics_predict_dual_yolov26(model_yolo_fire, model_yolo_smoke,
                                                                                   all_tiles, tiles_data,
                                                                                   conf_fire=CONFIDENCE_THRESHOLD_FIRE,
                                                                                   conf_smoke=CONFIDENCE_THRESHOLD_SMOKE)
            # print(detections)
            print("boxes", boxes)
            detections = []

            for box, score, label, pano in zip(boxes, scores, labels, pano_ids):
                detections.append({
                    "panorama_idx": int(pano.item()),
                    "bbox": box.tolist(),
                    "class": int(label.item()),
                    "confidence": float(score.item())
                })
            front_events = []
            rear_events = []

            event_no_front = 1
            event_no_rear = 1
            for det in detections:

                pano_idx = det["panorama_idx"]
                x1, y1, x2, y2 = det["bbox"]

                type_of_event = det["class"]

                print("type_of_event", type_of_event)
                print("checking")
                if type_of_event == 1 and det['confidence'] < CONFIDENCE_THRESHOLD_SMOKE:
                    print(1, det['confidence'])
                    continue

                type_of_event = CLASS_MAP_FIRE_SMOKE.get(
                    type_of_event,
                    "unknown"
                )

                # scale_x = 2000/8000
                # scale_y = 270/1080
                # x1 = x1*scale_x
                # x2 = x2*scale_x
                # y1 = y1*scale_y
                # y2 = y2*scale_y

                print(
                    "det['bbox']",
                    det['bbox'],
                    x1, y1, x2, y2, det['confidence']
                )

                event_dict = {
                    "event_no": (
                        event_no_front
                        if pano_idx == 0
                        else event_no_rear
                    ),
                    "xmin": int(x1),
                    "ymin": int(y1),
                    "xmax": int(x2),
                    "ymax": int(y2),
                    "type_of_event": type_of_event
                }

                if pano_idx == 0:
                    front_events.append(event_dict)
                    event_no_front += 1

                elif pano_idx == 1:
                    rear_events.append(event_dict)
                    event_no_rear += 1

            with ANALYTICS_LOCK:

                entry = ANALYTICS_RESULT.setdefault(
                    frame_id,
                    {"front": [], "rear": []}
                )

                entry["front"].extend(front_events)
                entry["rear"].extend(rear_events)
            print("data fire smoke appended")
            # print("image_fire_detected",time.time())

        eventlet.sleep(0.01)


def video_analytics_human_conflict():
    print("[HUMAN_CONFLICT] thread started")

    while True:

        try:

            image_front, image_back, frame_id = (
                analytics_queue_human_conflict.get(
                    timeout=0.1
                )
            )

        except Empty:

            continue

        print(image_front.shape)
        print(image_back.shape)
        top = image_front[:1080, :, :]
        bottom = image_front[1080:, :, :]

        image_front = np.concatenate([top, bottom], axis=1)

        top = image_back[:1080, :, :]
        bottom = image_back[1080:, :, :]

        image_back = np.concatenate([top, bottom], axis=1)

        print(image_front.shape)  # (1080, 7680, 3)
        print(image_back.shape)  # (1080, 7680, 3)

        front_enabled = EVENT_STATE[
            "/front"
        ]["human_conflict"]

        rear_enabled = EVENT_STATE[
            "/rear"
        ]["human_conflict"]

        front_events = []

        rear_events = []

        # =====================================================
        # FRONT
        # =====================================================
        # =====================================================
        # FRONT
        # =====================================================
        if front_enabled:

            for zone in range(NUM_ZONES):

                x1 = zone * ZONE_W
                x2 = x1 + ZONE_W

                zone_frame = image_front[:, x1:x2, :]

                FRONT_CONFLICT_BUFFERS[zone].append(zone_frame)

                if len(FRONT_CONFLICT_BUFFERS[zone]) == human_conflict_detector.clip_len:

                    fight, confidence = human_conflict_detector.predict(
                        FRONT_CONFLICT_BUFFERS[zone]
                    )

                    print(
                        f"[FRONT][ZONE {zone}] "
                        f"Fight={fight}, "
                        f"Conf={confidence:.3f}"
                    )

                    if fight:
                        front_events.append({

                            "event_no": zone + 1,

                            "zone": zone,

                            "type_of_event":
                                CLASS_MAP_HUMAN_CONFLICT["violence"],

                            "confidence":
                                round(confidence, 3)

                        })

        # =====================================================
        # REAR
        # =====================================================
        # =====================================================
        # REAR
        # =====================================================
        if rear_enabled:

            for zone in range(NUM_ZONES):

                x1 = zone * ZONE_W
                x2 = x1 + ZONE_W

                zone_frame = image_back[:, x1:x2, :]

                REAR_CONFLICT_BUFFERS[zone].append(zone_frame)

                if len(REAR_CONFLICT_BUFFERS[zone]) == human_conflict_detector.clip_len:

                    fight, confidence = human_conflict_detector.predict(
                        REAR_CONFLICT_BUFFERS[zone]
                    )

                    print(
                        f"[REAR][ZONE {zone}] "
                        f"Fight={fight}, "
                        f"Conf={confidence:.3f}"
                    )

                    if fight:
                        rear_events.append({

                            "event_no": zone + 1,

                            "zone": zone,

                            "type_of_event":
                                CLASS_MAP_HUMAN_CONFLICT["violence"],

                            "confidence":
                                round(confidence, 3)

                        })
        # =====================================================
        # STORE RESULTS
        # =====================================================
        if front_events or rear_events:
            with ANALYTICS_LOCK:
                entry = ANALYTICS_RESULT.setdefault(
                    frame_id,
                    {
                        "front": [],
                        "rear": []
                    }
                )

                entry["front"].extend(
                    front_events
                )

                entry["rear"].extend(
                    rear_events
                )

            print(
                "[HUMAN_CONFLICT] "
                "Event appended"
            )

        eventlet.sleep(0.001)


def warmup_models():
    print("[WARMUP] Starting...")

    dummy_CCD = torch.zeros(
        (8, 3, 1120, 1120),
        dtype=torch.float16,
        device='cuda'
    )
    dummy_IR = torch.zeros(
        (14, 3, 560, 560),
        dtype=torch.float16,
        device='cuda'
    )

    # ---- detection warmup ----
    # for _ in range(3):
    #     # num_classes, outputs = detectionModelCCD.predict(dummy_CCD)
    #     # num_classes, outputs = detectionModelIR.predict(dummy_IR)

    # ---- ROI + classification warmup ----

    dummy_boxes = torch.tensor(
        [[0.0, 0.0, 100.0, 100.0]],
        dtype=torch.float32,
        device='cuda'
    )

    # roi_fused expects Byte / uint8 tile IDs
    dummy_tile = torch.tensor(
        [0],
        dtype=torch.uint8,
        device='cuda'
    )

    for i in range(3):
        crops = roi_fused.roi_align(
            dummy_CCD,
            dummy_boxes,
            dummy_tile
        )

        print(
            f"[WARMUP] Iteration {i + 1} | "
            f"Crop shape: {crops.shape}"
        )

        # _ = classificationModelCCD.infer_patch_batch(crops)
        # _ = classificationModelIR.infer_patch_batch(crops)

    torch.cuda.synchronize()

    print("[WARMUP] Done ✅")


import os
import cv2
import numpy as np
import torch

DEBUG_SAVE_DIR = "./debug_bbox_images"
os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)


def save_combined_pano(
        cpu_buf3,
        cpu_buf4,
        targets_front,
        targets_back,
        frame_id
):
    # Convert CPU tensors to NumPy arrays
    front = cpu_buf3.numpy().copy()
    back = cpu_buf4.numpy().copy()

    # Draw boxes in the coordinate system of the buffers.
    # This assumes target coordinates already match these buffers.
    front = draw_bboxes(front, targets_front)
    back = draw_bboxes(back, targets_back)

    # Front on top, back on bottom
    combined = np.vstack((front, back))

    save_path = os.path.join(
        DEBUG_SAVE_DIR,
        f"combined_{frame_id:06d}.jpg"
    )

    cv2.imwrite(save_path, combined)

    print(
        f"[DEBUG IMAGE SAVED] {save_path} "
        f"Shape: {combined.shape}"
    )


def gpu_inference_thread(model1, sio, cam_CCD_FRONT, cam_CCD_REAR, cam_IR_FRONT, cam_IR_REAR):
    # RESOLUTION = 560
    TILE_W = 1120
    STEP = 1120
    MAX_BOX = 256
    cpu_buf1 = torch.empty((2160, 3840, 3), dtype=torch.uint8, pin_memory=True)
    cpu_buf2 = torch.empty((2160, 3840, 3), dtype=torch.uint8, pin_memory=True)
    cpu_buf3 = torch.empty((270, 2000, 3), dtype=torch.uint8, pin_memory=True)
    cpu_buf4 = torch.empty((270, 2000, 3), dtype=torch.uint8, pin_memory=True)
    if sh.CAMERA_MODE == 1:
        prev_tiles1 = torch.empty(
            (8, 3, 1120, 1120),
            dtype=torch.float16,
            device='cuda'
        )
    if sh.CAMERA_MODE == 0:
        prev_tiles11 = torch.empty(
            (14, 3, 560, 560),
            dtype=torch.float16,
            device='cuda'
        )

    stream_preprocess1 = torch.cuda.Stream()
    stream_preprocess2 = torch.cuda.Stream()
    stream_preprocess3 = torch.cuda.Stream()
    stream_inference1 = torch.cuda.Stream()
    stream_resize = torch.cuda.Stream()
    stream_cls = torch.cuda.Stream()
    stream_split1 = torch.cuda.Stream()
    stream_split2 = torch.cuda.Stream()
    stream_d2h = torch.cuda.Stream()
    stream_tile_merge = torch.cuda.Stream()
    stream_fire_smoke = torch.cuda.Stream()
    stream_human_conflict = torch.cuda.Stream()
    # stream_panorama = torch.cuda.Stream()

    print("[GPU] CUDA worker started")
    frame_id1 = 0
    frame_id2 = 0

    warmup_models()

    while True:
        if sh.CAMERA_MODE == 1:
            # frame1, ts1, seq1 = cam_CCD_FRONT.get_latest()
            # frame2, ts2, seq2 = cam_CCD_REAR.get_latest()
            frame1 = cam_CCD_FRONT.get_latest_bgr()
            frame2 = cam_CCD_REAR.get_latest_bgr()
        elif sh.CAMERA_MODE == 0:
            # frame1, ts1, seq1 = cam_IR_FRONT.get_latest()
            # frame2, ts2, seq2 = cam_IR_REAR.get_latest()
            frame1 = cam_IR_FRONT.get_latest_bgr()
            frame2 = cam_IR_REAR.get_latest_bgr()

        if frame1 is None or frame2 is None:  # first frame not decoded yet
            continue

        # ---------------------------------------------------------
        # Only frame1 available -> create black frame for frame2
        # ---------------------------------------------------------
        if frame1 is not None and frame2 is None:

            frame2 = np.zeros_like(frame1)

            print("[CAMERA] Stream 2 unavailable -> using black frame")


        # ---------------------------------------------------------
        # Only frame2 available -> create black frame for frame1
        # ---------------------------------------------------------
        elif frame1 is None and frame2 is not None:

            frame1 = np.zeros_like(frame2)

            print("[CAMERA] Stream 1 unavailable -> using black frame")

        # At this point BOTH frame1 and frame2 always exist
        # ---------------------------------------------------------

        print("frame1:", frame1.shape)
        print("frame2:", frame2.shape)

        # ---- H2D ----
        image1 = cv2.resize(frame1, (3840, 2160))
        image2 = cv2.resize(frame2, (3840, 2160))
        cpu_buf1.copy_(torch.from_numpy(image1), non_blocking=True)
        cpu_buf2.copy_(torch.from_numpy(image2), non_blocking=True)

        gpu_raw1 = cpu_buf1.to("cuda", non_blocking=True)
        gpu_raw2 = cpu_buf2.to("cuda", non_blocking=True)

        # # forward same frames to analytics
        # try:
        #     # analytics_queue_fire_smoke.put_nowait((image1, image2,frame_id1))
        #     analytics_queue_fire_smoke.put_nowait((gpu_raw1, gpu_raw2, frame_id1))
        # except:
        #     try:
        #         analytics_queue_fire_smoke.get_nowait()
        #         # analytics_queue_fire_smoke.put_nowait((image1, image2,frame_id1))
        #         analytics_queue_fire_smoke.put_nowait((gpu_raw1, gpu_raw2, frame_id1))
        #     except:
        #         pass
        # try:
        #     analytics_queue_human_conflict.put_nowait((image1, image2, frame_id1))
        # except:
        #     try:
        #         analytics_queue_human_conflict.get_nowait()
        #         analytics_queue_human_conflict.put_nowait((image1, image2, frame_id1))
        #     except:
        #         pass

        # ---- SPLIT ----
        main_stream = torch.cuda.current_stream()

        stream_split1.wait_stream(main_stream)
        stream_split2.wait_stream(main_stream)

        with torch.cuda.stream(stream_split1):
            gpu_proc1 = split_concat.run(gpu_raw1)

        with torch.cuda.stream(stream_split2):
            gpu_proc2 = split_concat.run(gpu_raw2)

        # forward same frames to analytics
        try:
            # analytics_queue_fire_smoke.put_nowait((image1, image2,frame_id1))
            analytics_queue_fire_smoke.put_nowait((gpu_proc1, gpu_proc2, frame_id1))
        except:
            try:
                analytics_queue_fire_smoke.get_nowait()
                # analytics_queue_fire_smoke.put_nowait((image1, image2,frame_id1))
                analytics_queue_fire_smoke.put_nowait((gpu_proc1, gpu_proc2, frame_id1))
            except:
                pass
        try:
            analytics_queue_human_conflict.put_nowait((image1, image2, frame_id1))
        except:
            try:
                analytics_queue_human_conflict.get_nowait()
                analytics_queue_human_conflict.put_nowait((image1, image2, frame_id1))
            except:
                pass

        # ---- PREPROCESS + RESIZE (parallel 🚀) ----
        stream_preprocess1.wait_stream(stream_split1)
        stream_preprocess2.wait_stream(stream_split2)
        stream_resize.wait_stream(stream_split1)
        stream_resize.wait_stream(stream_split2)

        with torch.cuda.stream(stream_preprocess1):

            img1 = gpu_proc1.permute(2, 0, 1).unsqueeze(0)
            if sh.CAMERA_MODE == 1:
                tiles1 = fused_preprocess_large.run(img1.permute(0, 2, 3, 1))
            if sh.CAMERA_MODE == 0:
                tiles1 = fused_preprocess_new.run(img1.permute(0, 2, 3, 1))

        with torch.cuda.stream(stream_preprocess2):
            img2 = gpu_proc2.permute(2, 0, 1).unsqueeze(0)
            if sh.CAMERA_MODE == 1:
                tiles2 = fused_preprocess_large.run(img2.permute(0, 2, 3, 1))
            if sh.CAMERA_MODE == 0:
                tiles2 = fused_preprocess_new.run(img2.permute(0, 2, 3, 1))

        with torch.cuda.stream(stream_resize):
            small1_gpu, small2_gpu = pano_resize_encoding_optimize_new.run(
                gpu_proc1, gpu_proc2
            )

        # ---- D2H (separate stream 🚀) ----
        stream_d2h.wait_stream(stream_resize)

        with torch.cuda.stream(stream_d2h):
            cpu_buf3.copy_(small1_gpu, non_blocking=True)
            cpu_buf4.copy_(small2_gpu, non_blocking=True)

        # ---- TILE MERGE ----
        stream_tile_merge.wait_stream(stream_preprocess1)
        stream_tile_merge.wait_stream(stream_preprocess2)

        with torch.cuda.stream(stream_tile_merge):
            if sh.CAMERA_MODE == 1:
                prev_tiles1[:4].copy_(tiles1, non_blocking=True)
                prev_tiles1[4:8].copy_(tiles2, non_blocking=True)
            if sh.CAMERA_MODE == 0:
                prev_tiles11[:7].copy_(tiles1, non_blocking=True)
                prev_tiles11[7:].copy_(tiles2, non_blocking=True)

        with torch.cuda.stream(stream_preprocess3):
            if sh.CAMERA_MODE == 1:
                TILE_W = 1920
                NUM_TILES = 4
            if sh.CAMERA_MODE == 0:
                TILE_W = 1120
                NUM_TILES = 7

            assert gpu_proc1.shape == gpu_proc2.shape
            assert gpu_proc1.shape[1] == 7680

            gpu_proc1 = gpu_proc1.permute(2, 0, 1).contiguous()
            gpu_proc2 = gpu_proc2.permute(2, 0, 1).contiguous()

            # BGR->RGB
            gpu_proc1 = gpu_proc1[[2, 1, 0], :, :]
            gpu_proc2 = gpu_proc2[[2, 1, 0], :, :]

            if sh.CAMERA_MODE == 0:
                gpu_proc1 = F.pad(gpu_proc1, (0, 160, 0, 0))
                gpu_proc2 = F.pad(gpu_proc2, (0, 160, 0, 0))

            cls_tiles1 = (
                gpu_proc1
                    .view(3, 1080, NUM_TILES, TILE_W)
                    .permute(2, 0, 1, 3)
                    .contiguous()
            )

            cls_tiles2 = (
                gpu_proc2
                    .view(3, 1080, NUM_TILES, TILE_W)
                    .permute(2, 0, 1, 3)
                    .contiguous()
            )
            cls_tiles = torch.cat((cls_tiles1, cls_tiles2), dim=0)
            print(cls_tiles.shape)

        # ---------------------------------------------------
        # FLAGS
        # ---------------------------------------------------
        det_front_on = DETECTION_STATE["/front"]
        det_rear_on = DETECTION_STATE["/rear"]
        run_detection = det_front_on or det_rear_on
        cls_front_on = CLASSIFICATION_STATE["/front"]
        cls_rear_on = CLASSIFICATION_STATE["/rear"]
        run_classification = cls_front_on or cls_rear_on
        front_on_fire = EVENT_STATE["/front"]['fire']
        rear_on_fire = EVENT_STATE["/rear"]['fire']
        front_on_smoke = EVENT_STATE["/front"]['smoke']
        rear_on_smoke = EVENT_STATE["/rear"]['smoke']
        run_fire_smoke = front_on_fire or front_on_smoke
        front_on_human_conflict = EVENT_STATE["/front"]['human_conflict']
        rear_on_human_conflict = EVENT_STATE["/rear"]['human_conflict']
        run_human_conflict = front_on_human_conflict or rear_on_human_conflict
        run_detection = True
        run_classification = True
        # run_human_conflict = True
        # run_fire_smoke = True

        # if front_on_fire and rear_on_fire and front_on_smoke and rear_on_smoke:
        #     print("fire/smoke analytics running")
        #     print("image_for_fire", time.time())

        # ---------------------------------------------------
        # INIT
        # ---------------------------------------------------
        primaryModelOutputs = None
        class_id = None

        # ---------------------------------------------------
        # WAIT FOR PREPROCESS
        # ---------------------------------------------------
        stream_inference1.wait_stream(stream_tile_merge)

        # if run_fire_smoke:
        #     with torch.cuda.stream(stream_fire_smoke):
        #         logger.debug("run_smoke: %s", run_fire_smoke)
        #         tiles1_fire = fire_smoke_preprocess.run(gpu_raw1)
        #         tiles2_fire = fire_smoke_preprocess.run(gpu_raw2)
        #         all_tiles = torch.cat([tiles1_fire, tiles2_fire], dim=0)
        #         d_boxes, d_scores, d_labels, d_pano  = video_analytics_predict_dual_yolov5(model_yolo_fire, model_yolo_smoke, all_tiles,offsets_gpu,       # precomputed
        #                                                         pano_ids_gpu,
        #                                                           conf_fire=CONFIDENCE_THRESHOLD_FIRE,
        #                                                          conf_smoke=CONFIDENCE_THRESHOLD_SMOKE,
        #                                                          )
        #         # here we are getting bbox and class of fire and smoke now send it to cpu and emit\
        # if run_human_conflict:
        #     with torch.cuda.stream(stream_human_conflict):
        #         logger.debug("run_human_conflict: %s", run_human_conflict)
        #         tiles1_fire = fire_smoke_preprocess.run(gpu_raw1)
        #         tiles2_fire = fire_smoke_preprocess.run(gpu_raw2)
        #         all_tiles = torch.cat([tiles1_fire, tiles2_fire], dim=0)
        #         detections = video_analytics_predict_yolov8(model_yolo_human_conflict, all_tiles,offsets_gpu,pano_ids_gpu,
        #                                                     conf_th=CONFIDENCE_THRESHOLD_FIRE)

        # ---------------------------------------------------
        # DETECTION + ROI
        # ---------------------------------------------------
        if run_detection:
            with torch.cuda.stream(stream_inference1):
                if sh.CAMERA_MODE == 1:
                    num_classes, primaryModelOutputs = detectionModelCCD.predict(prev_tiles1)
                    CLS_THRES = classificationModelCCD.CLS_THRES
                elif sh.CAMERA_MODE == 0:
                    num_classes, primaryModelOutputs = detectionModelIR.predict(prev_tiles11)
                    # CLS_THRES = classificationModelIR.CLS_THRES

                CLS_THRES = classificationModelCCD.CLS_THRES
                MAX_BOX = 256

                # ---- Python extraction (unavoidable small sync) ----
                boxes_list = [det.xyxy for det in primaryModelOutputs]
                conf_list = [det.confidence for det in primaryModelOutputs]

                # ---- CPU metadata ----
                tile_sizes_cpu = torch.tensor(
                    [b.shape[0] for b in boxes_list],
                    dtype=torch.int32
                )
                total_n = int(tile_sizes_cpu.sum())

                # ---- Move ONCE to GPU (important fix) ----
                tile_sizes = tile_sizes_cpu.to('cuda', non_blocking=True)

                # ---- flatten (GPU) ----
                all_boxes = torch.cat(boxes_list, dim=0)
                all_conf = torch.cat(conf_list, dim=0)

                # ---- tile ids (GPU, no CPU mixing) ----
                tile_ids = torch.repeat_interleave(
                    torch.arange(len(boxes_list), device='cuda', dtype=torch.int32),
                    tile_sizes
                ).to(torch.uint8)

                # ---- det ids (fully GPU, no hidden sync) ----
                global_ids = torch.arange(total_n, device='cuda', dtype=torch.int32)

                tile_offsets = torch.repeat_interleave(
                    torch.cumsum(
                        torch.cat([
                            torch.zeros(1, device='cuda', dtype=torch.int32),
                            tile_sizes[:-1]  # ✅ already on GPU
                        ]),
                        dim=0
                    ),
                    tile_sizes
                )

                det_ids = (global_ids - tile_offsets).to(torch.uint8)

                # ---- filter + cap (GPU only) ----
                mask = all_conf > CLS_THRES

                d_roi_xyxy = all_boxes[mask][:MAX_BOX]
                d_roi_tile = tile_ids[mask][:MAX_BOX]
                d_roi_det = det_ids[mask][:MAX_BOX]

                # ---- ROI ALIGN (always call, no branch) ----
                # crops = roi_fused.roi_align(prev_tiles1, d_roi_xyxy, d_roi_tile)

                crop_list = []

                for i in range(d_roi_xyxy.shape[0]):
                    tile_idx = int(d_roi_tile[i])

                    x1, y1, x2, y2 = d_roi_xyxy[i]

                    if sh.CAMERA_MODE == 1:
                        x1 = max(0, int(x1.item()))
                        y1 = max(0, int(y1.item()))
                        x2 = min(1920, int(x2.item()))
                        y2 = min(1080, int(y2.item()))

                    if sh.CAMERA_MODE == 0:
                        x1 = max(0, int(x1.item()))
                        y1 = max(0, int(y1.item()))
                        x2 = min(1120, int(x2.item()))
                        y2 = min(1080, int(y2.item()))

                    box_width = x2 - x1
                    box_height = y2 - y1

                    if box_width <= s.CLASSIFICATION_WIDTH_LIMIT or box_height <= s.CLASSIFICATION_HEIGHT_LIMIT:
                        print(f"Discarded detection {i}: object too small")
                        continue

                    crop = cls_tiles[
                           tile_idx,
                           :,
                           y1:y2,
                           x1:x2
                           ]
                    crop = resize_with_letterbox(
                        crop,
                        final_size=384,
                        pad_value=128
                    )

                    crop_list.append(crop)
                    # print("crop",crop.shape)
                    # # # ---------------------------------------------------
                    # # # VISUALIZE CLASSIFICATION INPUT
                    # # # ---------------------------------------------------
                    # #
                    # crop_vis = crop.detach().cpu().numpy()
                    #
                    # # CHW -> HWC
                    # if crop_vis.ndim == 3:
                    #     crop_vis = np.transpose(crop_vis, (1, 2, 0))
                    #
                    # # Handle normalized [0, 1] images
                    # print("crop_vis.max()",crop_vis.max())
                    # if crop_vis.max() <= 1.0:
                    #     print("normalized image")
                    #     crop_vis = crop_vis * 255.0
                    #
                    # crop_vis = np.clip(crop_vis, 0, 255).astype(np.uint8)
                    #
                    # # RGB -> BGR
                    # crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR)
                    #
                    # # Add useful debugging information
                    # cv2.putText(
                    #     crop_vis,
                    #     f"tile={tile_idx} det={i} bbox=({x1},{y1},{x2},{y2})",
                    #     (5, 20),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.45,
                    #     (0, 255, 0),
                    #     1,
                    #     cv2.LINE_AA
                    # )
                    # # Save the crop
                    # save_path = f"./detection_crops_CCD/crop_tile{tile_idx}_det{i}.jpg"
                    # cv2.imwrite(save_path, crop_vis)
                    # print(f"Saved: {save_path}")
                    #
                    # cv2.imshow("Classification Input", crop_vis)
                    #
                    # key = cv2.waitKey(0) & 0xFF
                    #
                    # if key == ord("q"):
                    #     cv2.destroyAllWindows()
                    #     break

        # ---------------------------------------------------
        # CLASSIFICATION + SCATTER
        # ---------------------------------------------------
        if run_detection and run_classification:
            stream_cls.wait_stream(stream_inference1)
            stream_cls.wait_stream(stream_preprocess3)
            with torch.cuda.stream(stream_cls):
                print(len(crop_list))
                crops_batch = torch.stack(crop_list, dim=0)
                print(crops_batch.shape)
                if sh.CAMERA_MODE == 1:
                    processed_scores, processed_class_ids = classificationModelCCD.infer_patch_batch(crops_batch)
                    print("processed_class_ids", processed_class_ids)
                if sh.CAMERA_MODE == 0:
                    processed_scores, processed_class_ids = classificationModelIR.infer_patch_batch(crops_batch)

            stream_inference1.wait_stream(stream_cls)
            with torch.cuda.stream(stream_inference1):
                if sh.CAMERA_MODE == 1:
                    class_id = scatter_fused.scatter_replace(
                        d_roi_tile, d_roi_det, processed_class_ids, 8, 128
                    )
                    print("class_id", class_id)
                if sh.CAMERA_MODE == 0:
                    class_id = scatter_fused.scatter_replace(
                        d_roi_tile, d_roi_det, processed_class_ids, 14, 128
                    )

        # ---------------------------------------------------
        # WRITEBACK
        # ---------------------------------------------------
        if run_detection and run_classification:
            if sh.CAMERA_MODE == 1:
                for t in range(8):
                    primaryModelOutputs[t].class_id = class_id[t]
            if sh.CAMERA_MODE == 0:
                for t in range(14):
                    primaryModelOutputs[t].class_id = class_id[t]

        # ---------------------------------------------------
        # FINAL OUTPUT
        # ---------------------------------------------------
        refined_outputs = primaryModelOutputs

        buffer_1 = jpeg_turbo.encode(cpu_buf3, quality=75)
        buffer_2 = jpeg_turbo.encode(cpu_buf4, quality=75)
        frame_in_bytes_front = base64.b64encode(buffer_1).decode()
        frame_in_bytes_back = base64.b64encode(buffer_2).decode()

        if run_detection:

            payload_front, payload_back = build_and_emit_front_back_payloads(refined_outputs,
                                                                             TILE_W,
                                                                             frame_id1,
                                                                             frame_in_bytes_front,
                                                                             frame_in_bytes_back, cpu_buf3, cpu_buf4)

        else:
            # build empty detection payloads (only image, no targets)
            print("EMPTY_DETECTION")
            payload_front = {
                "message_id": frame_id1,
                "sensor_id": 1,
                "panoramic_view_id": 1,
                "frame_id": frame_id1,
                "aice_epoch": int(time.time() * 1_000_000),
                "no_of_targets": 0,
                "targets": [],
                "image": frame_in_bytes_front
            }

            payload_back = {
                "message_id": frame_id1,
                "sensor_id": 1,
                "panoramic_view_id": 2,
                "frame_id": frame_id1,
                "aice_epoch": int(time.time() * 1_000_000),
                "no_of_targets": 0,
                "targets": [],
                "image": frame_in_bytes_back
            }

        # if ANALYTICS_STATE["/front"] or ANALYTICS_STATE["/rear"]:
        if (EVENT_STATE['/front']['fire'] and EVENT_STATE['/rear']['fire'] \
            and EVENT_STATE['/front']['smoke'] and EVENT_STATE['/rear']['smoke']) or (
        EVENT_STATE['/front']['human_conflict']):

            t0 = time.time()
            result = None

            # wait up to 100 ms for analytics of this frame
            while True:

                with ANALYTICS_LOCK:
                    result = ANALYTICS_RESULT.pop(frame_id1, None)

                if result is not None:
                    break

                if time.time() - t0 > 0.10:  # 100 ms safety timeout
                    break

                eventlet.sleep(0.001)

            if result is not None:
                if EVENT_STATE['/front']['fire'] and EVENT_STATE['/front']['smoke']:
                    payload_front["no_of_events"] = len(result["front"])
                    payload_front["events"] = result["front"]

                    # if EVENT_STATE['/rear']['fire'] and EVENT_STATE['/rear']['smoke']:
                    payload_back["no_of_events"] = len(result["rear"])
                    payload_back["events"] = result["rear"]

                elif EVENT_STATE['/front']['human_conflict']:
                    payload_front["no_of_events"] = len(result["front"])
                    payload_front["events"] = result["front"]
                    payload_back["no_of_events"] = len(result["rear"])
                    payload_back["events"] = result["rear"]

        sio.emit("frame", payload_front, namespace="/front")
        sio.emit("frame", payload_back, namespace="/rear")
        print("data emitted")
        # print("frame_emitted",time.time()-time_frame_start)
        eventlet.sleep(0)
        frame_id1 += 1
        frame_id2 += 1


if __name__ == '__main__':
    model1 = detectionModelCCD
    print("thread_starting")
    cam_CCD_FRONT = RTSPReader(url=RTSP_URL_DAY_FRONT, name="cam_CCD_FRONT", out_w=8000, out_h=1080,
                               source_mode=SOURCE_MODE, video_path=VIDEO_PATH_1, )
    cam_CCD_REAR = RTSPReader(url=RTSP_URL_DAY_REAR, name="cam_CCD_REAR", out_w=8000, out_h=1080,
                              source_mode=SOURCE_MODE, video_path=VIDEO_PATH_2, )

    cam_IR_FRONT = RTSPReader(url=RTSP_URL_IR_FRONT, name="cam_IR_FRONT", out_w=8000, out_h=1080,
                              source_mode=SOURCE_MODE, video_path=VIDEO_PATH_IR1, )
    cam_IR_REAR = RTSPReader(url=RTSP_URL_IR_REAR, name="cam_IR_REAR", out_w=8000, out_h=1080, source_mode=SOURCE_MODE,
                             video_path=VIDEO_PATH_IR2, )
    t_gpu1 = threading.Thread(target=video_analytics_fire_smoke, daemon=True)
    t_gpu1.start()

    t_gpu2 = threading.Thread(target=video_analytics_human_conflict, daemon=True)
    t_gpu2.start()

    eventlet.spawn(gpu_inference_thread, model1, sio, cam_CCD_FRONT, cam_CCD_REAR, cam_IR_FRONT, cam_IR_REAR)

    # t_gpu2 = threading.Thread(target=video_analytics_human_conflict, daemon=True)
    # t_gpu2.start()

    print("✅ Starting Eventlet server in MAIN thread")
    eventlet.wsgi.server(
        eventlet.listen(("0.0.0.0", 12010)),
        app)



