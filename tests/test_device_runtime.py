import numpy as np



def test_runtime_recognizes_after_hold_threshold():
    from app.device_runtime import DeviceRuntime

    class FakeClock:
        def __init__(self):
            self.now_ms = 0

        def now(self):
            return self.now_ms

    class FakeCamera:
        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeProcessor:
        def get_embedding(self, frame):
            return np.ones(4, dtype=np.float32)

        def compute_similarity(self, embedding, stored, threshold):
            return {
                "status": "ALLOWED",
                "name": "Naufal",
                "similarity": 0.91,
                "closest_match": "Naufal",
                "user_id": 1,
            }

    class FakeDB:
        def __init__(self):
            self.logged = []

        def get_all_embeddings(self):
            return [{"id": 1, "name": "Naufal", "embedding": np.ones(4, dtype=np.float32)}]

        def add_access_log(self, user_id, matched_name, status, similarity):
            self.logged.append((user_id, matched_name, status, similarity))

        def upsert_device_status(self, **kwargs):
            self.status = kwargs

    runtime = DeviceRuntime(
        camera=FakeCamera(),
        palm_processor=FakeProcessor(),
        db=FakeDB(),
        clock=FakeClock(),
        hold_ms=1000,
        cooldown_ms=3000,
    )

    runtime.clock.now_ms = 0
    runtime.tick()
    runtime.clock.now_ms = 1200
    runtime.tick()

    assert runtime.db.logged[0][2] == "ALLOWED"
