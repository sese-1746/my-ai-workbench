"""模块B 情感分析接口测试：健康检查、校验错误、任务生命周期、推理冒烟。"""

from __future__ import annotations

import unittest
import uuid

from tests.helpers import client, sentiment_csv, wait_job

REQUIRED_METADATA = (
    "id", "name", "algorithm", "history", "best_epoch", "test",
    "param_count", "train_seconds", "model_bytes", "data_fingerprint", "created_at",
)


class SentimentApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        response = client.post(
            "/api/v1/sentiment/train",
            files={"file": ("sentiment.csv", sentiment_csv(), "text/csv")},
            data={"name": "测试情感模型", "algorithm": "textcnn", "epochs": "1"},
        )
        assert response.status_code == 202, response.text
        cls.job_id = response.json()["id"]
        cls.job = wait_job("sentiment", cls.job_id)
        cls.model_id = cls.job["model"]["id"] if cls.job.get("model") else None

    # ---------- 健康检查 ----------
    def test_health_reports_module(self) -> None:
        body = client.get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("sentiment", body["models"])

    # ---------- 生命周期 ----------
    def test_job_completed_with_history(self) -> None:
        self.assertEqual(self.job["status"], "completed", self.job.get("error"))
        self.assertEqual(len(self.job["history"]), 1)
        entry = self.job["history"][0]
        self.assertIn("validation_accuracy", entry)
        self.assertIn("validation_macro_f1", entry)

    def test_metadata_has_confusion_matrix(self) -> None:
        models = client.get("/api/v1/sentiment/models").json()["data"]
        metadata = next(item for item in models if item["id"] == self.model_id)
        for key in REQUIRED_METADATA:
            self.assertIn(key, metadata)
        self.assertEqual(metadata["algorithm"], "textcnn")
        self.assertEqual(len(metadata["labels"]), 3)
        matrix = metadata["test"]["confusion_matrix"]
        self.assertEqual(len(matrix), 3)
        self.assertTrue(all(len(row) == 3 for row in matrix))
        self.assertEqual(sum(sum(row) for row in matrix), metadata["test_count"])

    # ---------- 校验错误 ----------
    def test_wrong_extension_415(self) -> None:
        response = client.post(
            "/api/v1/sentiment/train",
            files={"file": ("data.txt", sentiment_csv(), "text/plain")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 415)

    def test_oversized_upload_413(self) -> None:
        response = client.post(
            "/api/v1/sentiment/train",
            files={"file": ("big.csv", b"a" * (5 * 1024 * 1024 + 1), "text/csv")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 413)

    def test_missing_label_column_422(self) -> None:
        response = client.post(
            "/api/v1/sentiment/train",
            files={"file": ("bad.csv", b"text\n" + "很好\n".encode("utf-8") * 60, "text/csv")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 422)

    def test_text_and_file_conflict_422(self) -> None:
        response = client.post(
            "/api/v1/sentiment/train",
            files={"file": ("data.csv", sentiment_csv(), "text/csv")},
            data={"text": "text,label\n好,正面", "name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("其中一种", response.json()["detail"])

    def test_single_class_422(self) -> None:
        payload = ("text,label\n" + "这个产品还不错,中性\n" * 60).encode("utf-8")
        response = client.post(
            "/api/v1/sentiment/train",
            files={"file": ("one.csv", payload, "text/csv")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 422)

    def test_unknown_algorithm_422(self) -> None:
        response = client.post(
            "/api/v1/sentiment/train",
            files={"file": ("data.csv", sentiment_csv(), "text/csv")},
            data={"name": "x", "algorithm": "bert", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 422)

    # ---------- 推理 ----------
    def test_classify_returns_probabilities(self) -> None:
        response = client.post(
            "/api/v1/sentiment/classify",
            json={"model_id": self.model_id, "text": "质量很好，物流也很快，非常满意"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertIn(data["label"], ("正面", "负面", "中性"))
        total = sum(item["probability"] for item in data["probabilities"])
        self.assertAlmostEqual(total, 1.0, places=2)
        self.assertTrue(data["tokens"])
        self.assertTrue(response.json()["notice"])

    def test_classify_empty_text_422(self) -> None:
        response = client.post(
            "/api/v1/sentiment/classify", json={"model_id": self.model_id, "text": "   "}
        )
        self.assertEqual(response.status_code, 422)

    def test_classify_unknown_model_404(self) -> None:
        response = client.post(
            "/api/v1/sentiment/classify",
            json={"model_id": str(uuid.uuid4()), "text": "测试"},
        )
        self.assertEqual(response.status_code, 404)

    # ---------- 另一算法 + 删除 ----------
    def test_bigru_training_and_delete(self) -> None:
        response = client.post(
            "/api/v1/sentiment/train",
            files={"file": ("sentiment.csv", sentiment_csv(), "text/csv")},
            data={"name": "BiGRU 模型", "algorithm": "bigru", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 202, response.text)
        job = wait_job("sentiment", response.json()["id"])
        self.assertEqual(job["status"], "completed", job.get("error"))
        model_id = job["model"]["id"]
        self.assertEqual(job["model"]["algorithm"], "bigru")

        self.assertEqual(client.delete(f"/api/v1/sentiment/models/{model_id}").status_code, 200)
        self.assertEqual(client.delete(f"/api/v1/sentiment/models/{model_id}").status_code, 404)

    def test_zz_delete_main_model(self) -> None:
        self.assertEqual(client.delete(f"/api/v1/sentiment/models/{self.model_id}").status_code, 200)


if __name__ == "__main__":
    unittest.main()
