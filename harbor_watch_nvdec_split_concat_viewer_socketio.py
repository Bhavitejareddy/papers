"""
Harbor Watch - NVDEC + split/concat + local viewer + Socket.IO streaming.

This version intentionally does NOT load or run detection, classification,
fire/smoke analytics, human-conflict analytics, or any other ML model.

Pipeline:
RTSP/file -> NVDEC -> GPU frame -> split_concat.run() ->
    (1) CudaGLViewer window
    (2) JPEG/Base64 -> Socket.IO /front and /rear
"""

import os
import sys
import time
import base64
import threading
from datetime import datetime

import numpy as np
import torch
import cv2
import PyNvVideoCodec as nvc

# ---------------------------------------------------------------------
# CUDA / custom extension setup
# ---------------------------------------------------------------------
TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
if hasattr(os, "add_dll_directory") and os.path.isdir(TORCH_LIB):
    os.add_dll_directory(TORCH_LIB)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUSED_BACKEND = os.path.join(BASE_DIR, "fused_backend")
if hasattr(os, "add_dll_directory") and os.path.isdir(FUSED_BACKEND):
    os.add_dll_directory(FUSED_BACKEND)
if FUSED_BACKEND not in sys.path:
    sys.path.insert(0, FUSED_BACKEND)

import split_concat

# The OpenGL/CUDA viewer used in nvdec (2).py.
from cuda_gl_viewer import CudaGLViewer

# ---------------------------------------------------------------------
# Socket.IO server
# ---------------------------------------------------------------------
import eventlet
eventlet.monkey_patch()

import socketio

SOCKET_HOST = "0.0.0.0"
SOCKET_PORT = 12010

sio = socketio.Server(
    async_mode="eventlet",
    cors_allowed_origins="*"
)
socket_app = socketio.WSGIApp(sio)

@sio.event(namespace="/front")
def connect(sid, environ):
    print(f"[SOCKET] Front UI connected: {sid}")

@sio.event(namespace="/rear")
def connect(sid, environ):
    print(f"[SOCKET] Rear UI connected: {sid}")

@sio.event(namespace="/front")
def disconnect(sid):
    print(f"[SOCKET] Front UI disconnected: {sid}")

@sio.event(namespace="/rear")
def disconnect(sid):
    print(f"[SOCKET] Rear UI disconnected: {sid}")

# ---------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------
# SOURCE_MODE = 1: RTSP streams
# SOURCE_MODE = 0: local video files
SOURCE_MODE = int(os.getenv("SOURCE_MODE", "1"))

# CAMERA_MODE = 1: CCD/day cameras; CAMERA_MODE = 0: IR cameras
CAMERA_MODE = int(os.getenv("CAMERA_MODE", "1"))

RTSP_URL_DAY_FRONT = "rtsp://192.168.100.10:554/rtpstream/config1"
RTSP_URL_DAY_REAR  = "rtsp://192.168.100.10:554/rtpstream/config1"
RTSP_URL_IR_FRONT  = "rtsp://10.10.0.237:9000/video"
RTSP_URL_IR_REAR   = "rtsp://10.10.0.237:9000/video"

VIDEO_PATH_DAY_FRONT = r"E:\hw_latest_videos\cam_2_h264.mp4"
VIDEO_PATH_DAY_REAR  = r"E:\hw_latest_videos\cam_2_h264.mp4"
VIDEO_PATH_IR_FRONT  = r"E:\hw_latest_videos\cam_1_h264.mp4"
VIDEO_PATH_IR_REAR   = r"E:\hw_latest_videos\cam_1_h264.mp4"

# Existing UI frame dimensions in hw_v3.py were 2000 x 270.
UI_WIDTH = 2000
UI_HEIGHT = 270
JPEG_QUALITY = 75

# NVDEC packet wrapper. The NumPy backing buffer must remain alive
# until Decode() returns.
_PacketData = nvc._PyNvVideoCodec.PacketData

def make_packet(buf):
    packet_array = np.frombuffer(buf, dtype=np.uint8)
    packet = _PacketData()
    packet.bsl_data = packet_array.ctypes.data
    packet.bsl = packet_array.nbytes
    return packet, packet_array


class RTSPReader:
    """
    NVDEC-based reader. One thread per stream, frames stay on the GPU.

    get_latest()  -> (frame, timestamp, seq)
        frame: torch.cuda.Tensor, (H, W, 3) uint8, RGB   (None until first frame)
        timestamp: time.monotonic() at decode time
        seq: increments on every new frame
    """

    STALL_TIMEOUT = 15.0
    MAX_CONSECUTIVE_DECODE_ERRORS = 10

    def __init__(self, url, name="cam", reconnect_delay=2.0,
                 out_w=None, out_h=None,          # accepted for compatibility, ignored
                 source_mode=1,
                 video_path=None,
                 gpu_id=0):
        self.url = url
        self.name = name
        self.reconnect_delay = reconnect_delay
        self.source_mode = source_mode
        self.video_path = video_path
        self.gpu_id = gpu_id
        self.decode_frames = 0
        self.decode_start_time = time.perf_counter()
        self.decode_fps = 0.0

        if self.source_mode == 0 and not self.video_path:
            raise ValueError(f"[{self.name}] source_mode=0 but no video_path provided")

        self._cond = threading.Condition()
        self._latest = None
        self._latest_ts = 0.0
        self._seq = 0
        self._new_frame_event = threading.Event()   # kept for compatibility
        self._stop_event = threading.Event()

        # ── Stats ──────────────────────────────────────────────────────────
        self._stat_lock = threading.Lock()
        self._reset_stats(time.time())
        self._stat_report_interval = 10.0

        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name=f"reader-{name}")
        self._thread.start()

    # ── stats helpers ─────────────────────────────────────────────────────

    def _reset_stats(self, now):
        self._stat_frames_decoded = 0
        self._stat_decode_errors = 0
        self._stat_demux_errors = 0
        self._stat_packets_received = 0
        self._stat_t_decode_total = 0.0   # time in Decode() (approximate, GPU is async)
        self._stat_t_copy_total = 0.0     # time in GPU D2D clone
        self._stat_t_lock_total = 0.0     # time acquiring the frame lock
        self._stat_last_report = now

    def _avg_ms(self, total):
        n = self._stat_frames_decoded
        return (total / n * 1000) if n else 0.0

    def _maybe_report_stats(self):
        now = time.time()
        elapsed = now - self._stat_last_report
        if elapsed < self._stat_report_interval:
            return
        print(
            f"[{self.name}][stats] "
            f"interval={elapsed:.1f}s | "
            f"fps={self._stat_frames_decoded / elapsed:.1f} | "
            f"decoded={self._stat_frames_decoded} | "
            f"packets={self._stat_packets_received} | "
            f"decode_errors={self._stat_decode_errors} "
            f"demux_errors={self._stat_demux_errors} | "
            f"avg_decode={self._avg_ms(self._stat_t_decode_total):.2f}ms "
            f"avg_gpu_copy={self._avg_ms(self._stat_t_copy_total):.2f}ms "
            f"avg_lock_wait={self._avg_ms(self._stat_t_lock_total):.2f}ms"
        )
        self._reset_stats(now)

    def get_stats(self):
        """Snapshot of current counters, callable from any thread."""
        return {
            "frames_decoded":   self._stat_frames_decoded,
            "decode_errors":    self._stat_decode_errors,
            "demux_errors":     self._stat_demux_errors,
            "packets_received": self._stat_packets_received,
            "avg_decode_ms":    self._avg_ms(self._stat_t_decode_total),
            "avg_copy_ms":      self._avg_ms(self._stat_t_copy_total),
            "avg_lock_ms":      self._avg_ms(self._stat_t_lock_total),
        }

    # ── open ──────────────────────────────────────────────────────────────

    def _open(self):
        src = self.url if self.source_mode == 1 else self.video_path
        demux = nvc.CreateDemuxer(src)
        dec = nvc.CreateDecoder(
            gpuid=self.gpu_id,
            codec=demux.GetNvCodecId(),
            cudacontext=0,
            cudastream=0,
            usedevicememory=True,                       # frames stay on GPU
            outputColorType=nvc.OutputColorType.RGB,    # colour conversion on GPU
        )
        try:
            fps = float(demux.FrameRate()) or 25.0
        except Exception:
            fps = 25.0
        try:
            size = f"{demux.Width()}x{demux.Height()}"
        except Exception:
            size = "?"
        print(f"[{self.name}] {'RTSP' if self.source_mode == 1 else 'file'} "
              f"opened — {size} fps={fps:.2f}")
        return demux, dec, fps

    # ── decode loop (RTSP and file share it) ──────────────────────────────

    def _run(self, demux, dec, fps):
        is_file = self.source_mode == 0
        period = 1.0 / fps
        next_t = time.perf_counter()
        consecutive_failures = 0
        last_packet_time = time.time()
        it = iter(demux)

        print(f"[{self.name}] entering decode loop")

        while not self._stop_event.is_set():
            if not is_file and time.time() - last_packet_time > self.STALL_TIMEOUT:
                print(f"[{self.name}] stall detected, reconnecting…")
                return True

            try:
                packet = next(it)
            except StopIteration:
                print(f"[{self.name}] {'file ended, looping…' if is_file else 'stream ended'}")
                return True
            except Exception as e:
                self._stat_demux_errors += 1
                print(f"[{self.name}] demux error: {e}")
                return True

            if packet is None:
                continue

            last_packet_time = time.time()
            self._stat_packets_received += 1

            # ── timed: NVDEC decode ───────────────────────────────────────
            t0 = time.perf_counter()
            try:
                surfaces = dec.Decode(packet)
            except Exception as e:
                self._stat_decode_errors += 1
                consecutive_failures += 1
                print(f"[{self.name}] decode error ({consecutive_failures}): {e}")
                if consecutive_failures >= self.MAX_CONSECUTIVE_DECODE_ERRORS:
                    print(f"[{self.name}] too many decode errors, reconnecting…")
                    return True
                continue
            self._stat_t_decode_total += time.perf_counter() - t0

            for surf in surfaces:
                consecutive_failures = 0

                # ── timed: zero-copy view + one device-to-device copy ─────
                # (clone so the decoder can safely reuse its own buffer)
                t1 = time.perf_counter()
                frame = torch.from_dlpack(surf).clone()
                self._stat_t_copy_total += time.perf_counter() - t1

                # ── timed: publish ────────────────────────────────────────
                t2 = time.perf_counter()
                with self._cond:
                    self._stat_t_lock_total += time.perf_counter() - t2
                    self._latest = frame
                    self._latest_ts = time.monotonic()
                    self._seq += 1
                    self._cond.notify_all()
                self._new_frame_event.set()
                self._stat_frames_decoded += 1

                # pace file playback to its native frame rate (deadline-based)
                # if is_file:
                #     next_t += period
                #     delay = next_t - time.perf_counter()
                #     if delay > 0:
                #         time.sleep(delay)
                #     else:
                #         next_t = time.perf_counter()

            self._maybe_report_stats()

        return False

    # ── thread entry ──────────────────────────────────────────────────────

    def _reader_loop(self):
        print(f"[{self.name}] reader thread started "
              f"(mode={'RTSP' if self.source_mode == 1 else 'file'})")
        torch.cuda.set_device(self.gpu_id)

        while not self._stop_event.is_set():
            demux = dec = None
            should_reconnect = True
            try:
                demux, dec, fps = self._open()
                should_reconnect = self._run(demux, dec, fps)
            except Exception as e:
                print(f"[{self.name}] exception: {e}")
                should_reconnect = True
            finally:
                del dec, demux

            if not self._stop_event.is_set() and should_reconnect:
                delay = self.reconnect_delay if self.source_mode == 1 else 0.1
                print(f"[{self.name}] reopening in {delay}s…")
                time.sleep(delay)

        print(f"[{self.name}] reader stopped")

    # ── consumer API ──────────────────────────────────────────────────────

    def get_latest(self):
        """Returns (frame_gpu_tensor_or_None, timestamp, seq)."""
        with self._cond:
            return self._latest, self._latest_ts, self._seq

    def wait_new(self, last_seq, timeout=0.2):
        """Block until a frame newer than last_seq arrives, or timeout."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout)
            return self._latest, self._latest_ts, self._seq

    def get_latest_bgr(self):
        """Legacy bridge: CPU NumPy BGR (H, W, 3). Slow, use only while migrating."""
        frame, _, _ = self.get_latest()
        if frame is None:
            return None
        return frame.flip(-1).cpu().numpy()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)

# ---------------------------------------------------------------------
# Result generation: GPU panorama -> JPEG -> Base64 for the existing UI
# ---------------------------------------------------------------------
def processed_gpu_frame_to_jpeg_base64(gpu_frame):
    """
    Convert split_concat output to a JPEG Base64 string.

    split_concat output is expected to be a CUDA tensor containing an RGB
    panorama. This helper accepts HWC or CHW tensors and normalizes to
    2000x270 for compatibility with the existing Harbor Watch UI payload.
    """
    if not isinstance(gpu_frame, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, received {type(gpu_frame)}")

    tensor = gpu_frame.detach()

    if tensor.is_cuda:
        tensor = tensor.to("cpu", non_blocking=False)

    array = tensor.numpy()

    # Support HWC (H,W,C) and CHW (C,H,W).
    if array.ndim != 3:
        raise ValueError(f"Unexpected split_concat output shape: {array.shape}")

    if array.shape[-1] in (1, 3, 4):
        image_rgb = array
    elif array.shape[0] in (1, 3, 4):
        image_rgb = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"Cannot identify channel dimension: {array.shape}")

    if image_rgb.dtype != np.uint8:
        if np.issubdtype(image_rgb.dtype, np.floating):
            # If output is normalized [0,1], scale it to [0,255].
            if image_rgb.max(initial=0) <= 1.0:
                image_rgb = image_rgb * 255.0
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    if image_rgb.shape[2] == 4:
        image_rgb = image_rgb[:, :, :3]
    elif image_rgb.shape[2] == 1:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

    # Resize only the UI copy. The GPU panorama sent to CudaGLViewer
    # remains at the original split_concat output resolution.
    image_rgb = cv2.resize(
        image_rgb,
        (UI_WIDTH, UI_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # OpenCV JPEG encoding expects BGR.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode the panorama as JPEG")

    return base64.b64encode(encoded.tobytes()).decode("ascii")


def make_ui_payload(image_base64, frame_id, panoramic_view_id):
    """
    Keep the main fields used by the existing hw_v3.py UI payload.
    Detection is intentionally disabled, so targets is always empty.
    """
    return {
        "message_id": frame_id,
        "sensor_id": 1,
        "panoramic_view_id": panoramic_view_id,
        "frame_id": frame_id,
        "aice_epoch": int(time.time() * 1_000_000),
        "no_of_targets": 0,
        "targets": [],
        "image": image_base64
    }


# ---------------------------------------------------------------------
# Camera selection
# ---------------------------------------------------------------------
def get_active_camera_pair(cameras):
    if CAMERA_MODE == 1:
        return cameras["ccd_front"], cameras["ccd_rear"]
    if CAMERA_MODE == 0:
        return cameras["ir_front"], cameras["ir_rear"]
    raise ValueError("CAMERA_MODE must be 0 (IR) or 1 (CCD)")


# ---------------------------------------------------------------------
# Main frame loop: split/concat -> Python viewer + Socket.IO
# ---------------------------------------------------------------------
def run_video_pipeline(cameras, viewer):
    split_stream_front = torch.cuda.Stream()
    split_stream_rear = torch.cuda.Stream()

    front_ready = torch.cuda.Event()
    rear_ready = torch.cuda.Event()

    last_front_seq = -1
    frame_id = 0
    last_front_output = None
    last_rear_output = None
    last_log_time = 0.0

    print("[PIPELINE] Waiting for decoded camera frames...")

    while True:
        cam_front, cam_rear = get_active_camera_pair(cameras)

        # Wait briefly for a NEW front frame instead of busy-spinning.
        front_frame, front_ts, front_seq = cam_front.wait_new(
            last_front_seq,
            timeout=0.05
        )
        rear_frame, rear_ts, rear_seq = cam_rear.get_latest()

        if front_frame is None or rear_frame is None:
            eventlet.sleep(0.001)
            continue

        # Process a new pair only when the front camera advances.
        if front_seq != last_front_seq:
            last_front_seq = front_seq

            # Tell PyTorch that these tensors are consumed on these CUDA
            # streams, so their memory is not reused too early.
            front_frame.record_stream(split_stream_front)
            rear_frame.record_stream(split_stream_rear)

            with torch.cuda.stream(split_stream_front):
                last_front_output = split_concat.run(front_frame)
                front_ready.record(split_stream_front)

            with torch.cuda.stream(split_stream_rear):
                last_rear_output = split_concat.run(rear_frame)
                rear_ready.record(split_stream_rear)

            frame_id += 1

            if frame_id == 1:
                print(
                    "[PIPELINE] split_concat front:",
                    tuple(last_front_output.shape),
                    last_front_output.dtype,
                    last_front_output.device
                )
                print(
                    "[PIPELINE] split_concat rear:",
                    tuple(last_rear_output.shape),
                    last_rear_output.dtype,
                    last_rear_output.device
                )

            # Convert the two processed panoramas for the UI.
            # CUDA events ensure split_concat has completed before CPU reads.
            front_ready.synchronize()
            rear_ready.synchronize()

            front_b64 = processed_gpu_frame_to_jpeg_base64(last_front_output)
            rear_b64 = processed_gpu_frame_to_jpeg_base64(last_rear_output)

            front_payload = make_ui_payload(front_b64, frame_id, 1)
            rear_payload = make_ui_payload(rear_b64, frame_id, 2)

            # Send to the exact namespaces and event used by hw_v3.py.
            sio.emit("frame", front_payload, namespace="/front")
            sio.emit("frame", rear_payload, namespace="/rear")

            now = time.time()
            if now - last_log_time >= 5:
                print(
                    f"[PIPELINE] frame={frame_id} | "
                    f"front_seq={front_seq} rear_seq={rear_seq} | "
                    f"Socket.IO frames emitted"
                )
                last_log_time = now

        # Local Python/OpenGL viewer: front panorama on top, rear below.
        # The viewer is kept on the main thread, as required by many
        # OpenGL window systems.
        if last_front_output is not None and last_rear_output is not None:
            keep_running = viewer.show(
                last_front_output,
                last_rear_output,
                ready_top=front_ready,
                ready_bottom=rear_ready
            )
            if not keep_running:
                print("[VIEWER] Window closed by user.")
                break

        # Let Eventlet service Socket.IO connections and emits.
        eventlet.sleep(0)


def start_socket_server():
    print(f"[SOCKET] Listening on http://0.0.0.0:{SOCKET_PORT}")
    eventlet.wsgi.server(
        eventlet.listen((SOCKET_HOST, SOCKET_PORT)),
        socket_app,
        log_output=False
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. NVDEC pipeline requires an NVIDIA GPU.")

    cameras = {}
    viewer = None

    camera_configs = {
        "ccd_front": (RTSP_URL_DAY_FRONT, VIDEO_PATH_DAY_FRONT),
        "ccd_rear":  (RTSP_URL_DAY_REAR, VIDEO_PATH_DAY_REAR),
        "ir_front":  (RTSP_URL_IR_FRONT, VIDEO_PATH_IR_FRONT),
        "ir_rear":   (RTSP_URL_IR_REAR, VIDEO_PATH_IR_REAR),
    }

    try:
        # Start one NVDEC reader thread per camera.
        for name, (url, video_path) in camera_configs.items():
            cameras[name] = RTSPReader(
                url=url,
                name=name,
                source_mode=SOURCE_MODE,
                video_path=video_path,
                gpu_id=0
            )

        # Wait for the selected pair to produce first frames.
        active_front, active_rear = get_active_camera_pair(cameras)
        for camera in (active_front, active_rear):
            frame, timestamp, seq = camera.wait_new(last_seq=0, timeout=30.0)
            if frame is None:
                raise RuntimeError(f"[{camera.name}] No decoded frame within 30 seconds")
            print(
                f"[READY] {camera.name}: shape={tuple(frame.shape)}, "
                f"dtype={frame.dtype}, device={frame.device}"
            )

        # Start Socket.IO server in an Eventlet green thread.
        eventlet.spawn(start_socket_server)

        # OpenGL viewer must be created and driven on the main thread.
        viewer = CudaGLViewer(
            window_width=1600,
            window_height=900,
            title="Harbor Watch - Split/Concat Front + Rear"
        )

        run_video_pipeline(cameras, viewer)

    except KeyboardInterrupt:
        print("[MAIN] Interrupted by user.")
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception as exc:
                print("[VIEWER] Close error:", exc)

        for camera in cameras.values():
            try:
                camera.stop()
            except Exception as exc:
                print(f"[CLEANUP] {camera.name}: {exc}")

        cv2.destroyAllWindows()
        print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    main()
