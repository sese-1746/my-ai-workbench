"""模块A 房价预测接口测试：健康检查、校验错误、任务生命周期、推理冒烟。"""

from __future__ import annotations

import unittest
import uuid

from tests.helpers import client, house_csv, wait_job

REQUIRED_METADATA = (
    "id", "name", "algorithm", "history", "best_epoch", "test",
    "param_count", "train_seconds", "model_bytes", "data_fingerprint", "created_at",
)


class HouseApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        response = client.post(
            "/api/v1/house/train",
            files={"file": ("house.csv", house_csv(), "text/csv")},
            data={"name": "测试房价模型", "epochs": "2"},
        )
        assert response.status_code == 202, response.text
        cls.job_id = response.json()["id"]
        cls.job = wait_job("house", cls.job_id)
        cls.model_id = cls.job["model"]["id"] if cls.job.get("model") else None

    # ---------- 健康检查 ----------
    def test_health_reports_module(self) -> None:
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("house", body["models"])
        self.assertGreaterEqual(body["models"]["house"], 1)

    # ---------- 任务生命周期 ----------
    def test_job_reaches_completed(self) -> None:
        self.assertEqual(self.job["status"], "completed", self.job.get("error"))
        self.assertEqual(self.job["current_epoch"], 2)
        self.assertEqual(len(self.job["history"]), 2)
        self.assertEqual(self.job["history"][0]["epoch"], 1)
        self.assertIn("train_log_rmse", self.job["history"][0])
        self.assertIn("validation_log_rmse", self.job["history"][0])

    def test_job_unknown_returns_404(self) -> None:
        response = client.get(f"/api/v1/house/jobs/{uuid.uuid4()}")
        self.assertEqual(response.status_code, 404)

    def test_metadata_contains_required_fields(self) -> None:
        self.assertIsNotNone(self.model_id)
        models = client.get("/api/v1/house/models").json()["data"]
        metadata = next(item for item in models if item["id"] == self.model_id)
        for key in REQUIRED_METADATA:
            self.assertIn(key, metadata)
        self.assertEqual(metadata["name"], "测试房价模型")
        self.assertEqual(metadata["algorithm"], "mlp")
        self.assertIn("log_rmse", metadata["test"])
        self.assertIn("mae", metadata["test"])
        self.assertIn("r2", metadata["test"])
        self.assertEqual(metadata["test_count"] + metadata["train_count"] + metadata["validation_count"], 120)

    # ---------- 校验错误 ----------
    def test_wrong_extension_415(self) -> None:
        response = client.post(
            "/api/v1/house/train",
            files={"file": ("house.txt", house_csv(100), "text/plain")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 415)

    def test_oversized_upload_413(self) -> None:
        big = b"a" * (10 * 1024 * 1024 + 1)
        response = client.post(
            "/api/v1/house/train",
            files={"file": ("big.csv", big, "text/csv")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 413)

    def test_missing_target_column_422(self) -> None:
        payload = b"Id,Area\n" + b"\n".join(b"%d,%d" % (i, i + 60) for i in range(120))
        response = client.post(
            "/api/v1/house/train",
            files={"file": ("bad.csv", payload, "text/csv")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("SalePrice", response.json()["detail"])

    def test_too_few_rows_422(self) -> None:
        response = client.post(
            "/api/v1/house/train",
            files={"file": ("small.csv", house_csv(20), "text/csv")},
            data={"name": "x", "epochs": "1"},
        )
        self.assertEqual(response.status_code, 422)

    def test_epochs_out_of_range_422(self) -> None:
        response = client.post(
            "/api/v1/house/train",
            files={"file": ("house.csv", house_csv(), "text/csv")},
            data={"name": "x", "epochs": "0"},
        )
        self.assertEqual(response.status_code, 422)

    # ---------- 推理 ----------
    def test_predict_returns_price_and_notice(self) -> None:
        response = client.post(
            "/api/v1/house/predict",
            json={
                "model_id": self.model_id,
                "features": {"Area": 120, "Rooms": 3, "Age": 5, "Neighborhood": "A"},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertGreater(body["data"]["price"], 0)
        self.assertEqual(body["data"]["currency"], "USD")
        self.assertTrue(body["notice"])

    def test_predict_unknown_field_422(self) -> None:
        response = client.post(
            "/api/v1/house/predict",
            json={"model_id": self.model_id, "features": {"不存在": 1}},
        )
        self.assertEqual(response.status_code, 422)

    def test_predict_non_numeric_value_422(self) -> None:
        response = client.post(
            "/api/v1/house/predict",
            json={"model_id": self.model_id, "features": {"Area": "很大"}},
        )
        self.assertEqual(response.status_code, 422)

    def test_predict_unknown_model_404(self) -> None:
        response = client.post(
            "/api/v1/house/predict",
            json={"model_id": str(uuid.uuid4()), "features": {}},
        )
        self.assertEqual(response.status_code, 404)

    # ---------- 删除 ----------
    def test_zz_delete_model(self) -> None:
        response = client.delete(f"/api/v1/house/models/{self.model_id}")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(any(item["id"] == self.model_id for item in client.get("/api/v1/house/models").json()["data"]))
        again = client.delete(f"/api/v1/house/models/{self.model_id}")
        self.assertEqual(again.status_code, 404)


if __name__ == "__main__":
    unittest.main()
