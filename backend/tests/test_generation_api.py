"""模块C 文本生成接口测试：健康检查、校验错误、任务生命周期、推理冒烟。"""

from __future__ import annotations

import unittest
import uuid

from tests.helpers import client, generation_corpus, wait_job

REQUIRED_METADATA = (
    "id", "name", "algorithm", "history", "best_epoch", "test",
    "param_count", "train_seconds", "model_bytes", "data_fingerprint", "created_at",
)


class GenerationApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        response = client.post(
            "/api/v1/generation/train",
            files={"file": ("corpus.txt", generation_corpus(), "text/plain")},
            data={"name": "测试生成模型", "epochs": "1"},
        )
        assert response.status_code == 202, response.text
        cls.job_id = response.json()["id"]
        cls.job = wait_job("generation", cls.job_id)
        cls.model_id = cls.job["model"]["id"] if cls.job.get("model") else None

    def test_health_reports_module(self) -> None:
        body = client.get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("generation", body["models"])

    def test_job_completed_with_perplexity(self) -> None:
        self.assertEqual(self.job["status"], "completed", self.job.get("error"))
        self.assertEqual(self.job["current_epoch"], 1)
        self.assertEqual(len(self.job["history"]), 1)
        entry = self.job["history"][0]
        self.assertIn("validation_loss", entry)
        self.assertGreater(entry["validation_perplexity"], 0)

    def test_metadata_fields(self) -> None:
        models = client.get("/api/v1/generation/models").json()["data"]
        metadata = next(item for item in models if item["id"] == self.model_id)
        for key in REQUIRED_METADATA:
            self.assertIn(key, metadata)
        self.assertEqual(metadata["algorithm"], "char-transformer")
        self.assertLessEqual(metadata["vocab_size"], 3000)
        self.assertEqual(metadata["context"], 96)
        self.assertIn("perplexity", metadata["test"])
        self.assertGreater(metadata["param_count"], 0)

    # ---------- 校验错误 ----------
    def test_wrong_extension_415(self) -> None:
        response = client.post(
            "/api/v1/generation/train",
            files={"file": ("corpus.md", generation_corpus(), "text/markdown")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 415)

    def test_oversized_upload_413(self) -> None:
        response = client.post(
            "/api/v1/generation/train",
            files={"file": ("big.txt", b"a" * (2 * 1024 * 1024 + 1), "text/plain")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 413)

    def test_too_few_documents_422(self) -> None:
        response = client.post(
            "/api/v1/generation/train",
            files={"file": ("tiny.txt", "第一句话。\n第二句话。".encode("utf-8"), "text/plain")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 422)

    def test_text_and_file_conflict_422(self) -> None:
        response = client.post(
            "/api/v1/generation/train",
            files={"file": ("corpus.txt", generation_corpus(), "text/plain")},
            data={"text": "一些语料内容", "name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 422)

    def test_epochs_out_of_range_422(self) -> None:
        response = client.post(
            "/api/v1/generation/train",
            files={"file": ("corpus.txt", generation_corpus(), "text/plain")},
            data={"name": "x", "epochs": "21"},
        )
        self.assertEqual(response.status_code, 422)

    # ---------- 推理 ----------
    def test_generate_returns_text(self) -> None:
        response = client.post(
            "/api/v1/generation/generate",
            json={
                "model_id": self.model_id,
                "prompt": "人工智能正在",
                "max_new_tokens": 20,
                "temperature": 0.8,
                "top_k": 10,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertGreater(body["data"]["generated_chars"], 0)
        self.assertLessEqual(body["data"]["generated_chars"], 20)
        self.assertIn(body["data"]["text"], body["data"]["full_text"])
        self.assertTrue(body["notice"])

    def test_generate_temperature_out_of_range_422(self) -> None:
        response = client.post(
            "/api/v1/generation/generate",
            json={"model_id": self.model_id, "prompt": "你好", "temperature": 2.0},
        )
        self.assertEqual(response.status_code, 422)

    def test_generate_length_out_of_range_422(self) -> None:
        response = client.post(
            "/api/v1/generation/generate",
            json={"model_id": self.model_id, "prompt": "你好", "max_new_tokens": 5},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("max_new_tokens", response.json()["detail"])

    def test_generate_unknown_model_404(self) -> None:
        response = client.post(
            "/api/v1/generation/generate",
            json={"model_id": str(uuid.uuid4()), "prompt": "你好"},
        )
        self.assertEqual(response.status_code, 404)

    def test_zz_delete_model(self) -> None:
        self.assertEqual(client.delete(f"/api/v1/generation/models/{self.model_id}").status_code, 200)
        self.assertFalse(
            any(
                item["id"] == self.model_id
                for item in client.get("/api/v1/generation/models").json()["data"]
            )
        )


if __name__ == "__main__":
    unittest.main()
