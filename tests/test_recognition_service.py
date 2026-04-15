import logging
import numpy as np



def test_match_and_log_allowed_result(caplog):
    class FakeProcessor:
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
            self.logged = None

        def get_all_embeddings(self):
            return [{"id": 1, "name": "Naufal", "embedding": np.ones(4, dtype=np.float32)}]

        def add_access_log(self, user_id, matched_name, status, similarity):
            self.logged = (user_id, matched_name, status, similarity)

    from app.services.recognition_service import match_embedding_and_log

    caplog.set_level(logging.INFO, logger="palmgate")

    db = FakeDB()
    result = match_embedding_and_log(FakeProcessor(), db, np.ones(4, dtype=np.float32), 0.75)

    assert result["status"] == "ALLOWED"
    assert db.logged == (1, "Naufal", "ALLOWED", 0.91)
    assert "ALLOWED | user=Naufal | similarity=0.9100" in caplog.text
