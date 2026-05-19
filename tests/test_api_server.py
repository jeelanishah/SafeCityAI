from fastapi.testclient import TestClient

from api import server


class StubDetector:
    class_names = ["Helmet", "No_Helmet", "License_Plate"]

    def predict(self, image_path: str) -> list[dict]:
        return [
            {"class_name": "Helmet", "confidence": 0.91},
            {"class_name": "No_Helmet", "confidence": 0.88},
        ]

    def get_violations(self, detections: list[dict]) -> list[dict]:
        return [d for d in detections if d["class_name"] == "No_Helmet"]


def _client() -> TestClient:
    server.detector = StubDetector()
    return TestClient(server.app)


def test_health() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info() -> None:
    response = _client().get("/model-info")
    assert response.status_code == 200
    assert response.json()["class_names"] == StubDetector.class_names


def test_detect() -> None:
    response = _client().post(
        "/detect",
        files={"file": ("test.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["detection_count"] == 2
    assert body["violation_count"] == 1
    assert body["violations"][0]["class_name"] == "No_Helmet"


def test_results_lookup() -> None:
    server.results_cache["job-1"] = {"job_id": "job-1", "detection_count": 0, "violations": []}

    found = _client().get("/results/job-1")
    missing = _client().get("/results/unknown")

    assert found.status_code == 200
    assert found.json()["job_id"] == "job-1"
    assert missing.status_code == 404


def test_upload_video_and_cache(monkeypatch) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self._frames = [object(), object()]

        def isOpened(self) -> bool:
            return True

        def read(self):
            if self._frames:
                return True, self._frames.pop(0)
            return False, None

        def release(self) -> None:
            return None

    class FakeCV2:
        def VideoCapture(self, path: str) -> FakeCapture:
            return FakeCapture()

        def imwrite(self, path: str, frame: object) -> bool:
            with open(path, "wb") as file:
                file.write(b"frame")
            return True

    monkeypatch.setattr(server, "cv2", FakeCV2())
    response = _client().post(
        "/upload-video",
        files={"file": ("clip.mp4", b"fake-video-bytes", "video/mp4")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["processed_frames"] == 2
    assert body["job_id"] in server.results_cache
