import threading
import time

from app.camera import OpenCVCameraSource
from app.config import (
    CAMERA_DEVICE_INDEX,
    DEVICE_COOLDOWN_MS,
    DEVICE_FRAME_INTERVAL_MS,
    DEVICE_HOLD_MS,
    DEVICE_STATUS_HEARTBEAT_MS,
    SIMILARITY_THRESHOLD,
)
from app.services.recognition_service import match_embedding_and_log


class SystemClock:
    def now(self):
        return int(time.time() * 1000)


class DeviceRuntime:
    def __init__(
        self,
        camera,
        palm_processor,
        db,
        clock=None,
        hold_ms: int = DEVICE_HOLD_MS,
        cooldown_ms: int = DEVICE_COOLDOWN_MS,
        frame_interval_ms: int = DEVICE_FRAME_INTERVAL_MS,
        heartbeat_ms: int = DEVICE_STATUS_HEARTBEAT_MS,
        threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.camera = camera
        self.palm_processor = palm_processor
        self.db = db
        self.clock = clock or SystemClock()
        self.hold_ms = hold_ms
        self.cooldown_ms = cooldown_ms
        self.frame_interval_ms = frame_interval_ms
        self.heartbeat_ms = heartbeat_ms
        self.threshold = threshold
        self.hand_seen_since_ms = None
        self.cooldown_until_ms = 0
        self.last_heartbeat_ms = None
        self.last_recognition_at = None
        self._thread = None
        self._stop_event = threading.Event()

    def tick(self):
        now_ms = self.clock.now()

        if self.last_heartbeat_ms is None or now_ms - self.last_heartbeat_ms >= self.heartbeat_ms:
            self.db.upsert_device_status(
                worker_state="running",
                camera_connected=True,
                last_error=None,
                fps=(1000 / self.frame_interval_ms) if self.frame_interval_ms > 0 else None,
                last_inference_ms=None,
                last_recognition_at=self.last_recognition_at,
            )
            self.last_heartbeat_ms = now_ms

        if now_ms < self.cooldown_until_ms:
            return None

        frame = self.camera.read()
        embedding = self.palm_processor.get_embedding(frame)
        if embedding is None:
            self.hand_seen_since_ms = None
            return None

        if self.hand_seen_since_ms is None:
            self.hand_seen_since_ms = now_ms
            return None

        if now_ms - self.hand_seen_since_ms < self.hold_ms:
            return None

        result = match_embedding_and_log(self.palm_processor, self.db, embedding, self.threshold)
        self.last_recognition_at = str(now_ms)
        self.cooldown_until_ms = now_ms + self.cooldown_ms
        self.hand_seen_since_ms = None
        self.db.upsert_device_status(
            worker_state="running",
            camera_connected=True,
            last_error=None,
            fps=(1000 / self.frame_interval_ms) if self.frame_interval_ms > 0 else None,
            last_inference_ms=0.0,
            last_recognition_at=self.last_recognition_at,
        )
        return result

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:
                self.db.upsert_device_status(
                    worker_state="error",
                    camera_connected=False,
                    last_error=str(exc),
                    fps=None,
                    last_inference_ms=None,
                    last_recognition_at=self.last_recognition_at,
                )
            time.sleep(self.frame_interval_ms / 1000)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        close = getattr(self.camera, "close", None)
        if callable(close):
            close()


def build_device_runtime(palm_processor, db):
    camera = OpenCVCameraSource(CAMERA_DEVICE_INDEX)
    return DeviceRuntime(camera=camera, palm_processor=palm_processor, db=db)


if __name__ == "__main__":
    from app.config import DB_PATH
    from app.database import Database
    from app.palm_processor import PalmProcessor

    db = Database(DB_PATH)
    palm_processor = PalmProcessor()
    runtime = build_device_runtime(palm_processor, db)

    try:
        runtime.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()
        palm_processor.close()
        db.close()
