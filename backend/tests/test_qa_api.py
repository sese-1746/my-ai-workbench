"""模块D 问答检索接口测试：健康检查、校验错误、任务生命周期、推理冒烟。"""

from __future__ import annotations

import unittest
import uuid

from tests.helpers import client, qa_csv, wait_job

REQUIRED_METADATA = (
    "id", "name", "algorithm", "history", "best_epoch", "test",
    "param_count", "train_seconds", "model_bytes", "data_fingerprint", "created_at",
)


class QaApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        response = client.post(
            "/api/v1/qa/train",
            files={"file": ("qa.csv", qa_csv(), "text/csv")},
            data={"name": "测试问答模型", "epochs": "2"},
        )
        assert response.status_code == 202, response.text
        cls.job_id = response.json()["id"]
        cls.job = wait_job("qa", cls.job_id)
        cls.model_id = cls.job["model"]["id"] if cls.job.get("model") else None

    def test_health_reports_module(self) -> None:
        body = client.get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("qa", body["models"])

    def test_job_completed_with_config_search(self) -> None:
        self.assertEqual(self.job["status"], "completed", self.job.get("error"))
        self.assertEqual(len(self.job["history"]), 2)
        entry = self.job["history"][0]
        self.assertIn("config", entry)
        self.assertIn("validation_mrr", entry)

    def test_metadata_test_metrics(self) -> None:
        models = client.get("/api/v1/qa/models").json()["data"]
        metadata = next(item for item in models if item["id"] == self.model_id)
        for key in REQUIRED_METADATA:
            self.assertIn(key, metadata)
        self.assertEqual(metadata["algorithm"], "tfidf-dual")
        test = metadata["test"]
        for key in ("recall_at_1", "recall_at_3", "mrr", "confusion_matrix"):
            self.assertIn(key, test)
        self.assertEqual(len(test["confusion_matrix"]), 2)
        self.assertGreaterEqual(test["recall_at_3"], test["recall_at_1"])
        self.assertIn("best_config", metadata)

    # ---------- 校验错误 ----------
    def test_wrong_extension_415(self) -> None:
        response = client.post(
            "/api/v1/qa/train",
            files={"file": ("qa.xlsx", qa_csv(), "application/vnd.ms-excel")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 415)

    def test_oversized_upload_413(self) -> None:
        response = client.post(
            "/api/v1/qa/train",
            files={"file": ("big.csv", b"a" * (5 * 1024 * 1024 + 1), "text/csv")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 413)

    def test_too_few_pairs_422(self) -> None:
        payload = "question,answer\n你好,你好呀\n".encode("utf-8")
        response = client.post(
            "/api/v1/qa/train",
            files={"file": ("qa.csv", payload, "text/csv")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("问答对", response.json()["detail"])

    def test_missing_answer_column_422(self) -> None:
        payload = ("question\n" + "问题\n" * 20).encode("utf-8")
        response = client.post(
            "/api/v1/qa/train",
            files={"file": ("qa.csv", payload, "text/csv")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 422)

    def test_text_and_file_conflict_422(self) -> None:
        response = client.post(
            "/api/v1/qa/train",
            files={"file": ("qa.csv", qa_csv(), "text/csv")},
            data={"text": "question,answer\n你好,在的", "name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 422)

    # ---------- 推理 ----------
    def test_query_returns_ranked_answers(self) -> None:
        response = client.post(
            "/api/v1/qa/query",
            json={"model_id": self.model_id, "question": "路由器怎么重置", "top_k": 3},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        results = body["data"]["results"]
        self.assertEqual(len(results), 3)
        self.assertEqual([item["rank"] for item in results], [1, 2, 3])
        self.assertGreaterEqual(results[0]["score"], results[-1]["score"])
        self.assertIn("重置", results[0]["answer"])
        self.assertTrue(body["notice"])

    def test_query_top_k_out_of_range_422(self) -> None:
        response = client.post(
            "/api/v1/qa/query",
            json={"model_id": self.model_id, "question": "退款", "top_k": 20},
        )
        self.assertEqual(response.status_code, 422)

    def test_query_blank_question_422(self) -> None:
        response = client.post(
            "/api/v1/qa/query",
            json={"model_id": self.model_id, "question": "   ", "top_k": 3},
        )
        self.assertEqual(response.status_code, 422)

    def test_query_unknown_model_404(self) -> None:
        response = client.post(
            "/api/v1/qa/query",
            json={"model_id": str(uuid.uuid4()), "question": "退款", "top_k": 3},
        )
        self.assertEqual(response.status_code, 404)

    def test_zz_delete_model(self) -> None:
        self.assertEqual(client.delete(f"/api/v1/qa/models/{self.model_id}").status_code, 200)
        self.assertEqual(
            client.post(
                "/api/v1/qa/query",
                json={"model_id": self.model_id, "question": "退款", "top_k": 3},
            ).status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
