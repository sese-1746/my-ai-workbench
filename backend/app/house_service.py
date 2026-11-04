"""模块A 房价预测：表格回归 MLP。

管线：数值列均值填充 + 标准化，类别列独热；目标做 log 变换后再标准化；
``train / validation / holdout-test`` 按 60/20/20 划分，按验证集 Log RMSE 保存最佳权重。
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    SEED,
    DataError,
    JobRegistry,
    ModelStore,
    file_tree_bytes,
    sha256_hex,
    utc_now,
)

MAX_BYTES = 10 * 1024 * 1024
MAX_FEATURE_DIM = 5000
TARGET_COLUMN = "SalePrice"
WEIGHT_FILE = "weights.pt"
PREPROCESSOR_FILE = "preprocessor.joblib"


class HouseDataError(DataError):
    """房价数据或特征不合法。"""


def make_model(feature_count: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(feature_count, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    )


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transforms = []
    if numeric:
        transforms.append((
            "num",
            Pipeline([
                ("fillna", SimpleImputer(strategy="mean", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transforms.append((
            "cat",
            Pipeline([
                ("fillna", SimpleImputer(strategy="constant", fill_value="NaN", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transforms, sparse_threshold=0)


def read_dataset(raw: bytes) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """解析 CSV 并做结构性校验，返回 (特征, 目标, 数值列, 类别列)。"""
    if not raw:
        raise HouseDataError("CSV 文件为空，请重新选择文件")
    if len(raw) > MAX_BYTES:
        raise HouseDataError("CSV 不能超过 10 MB")
    try:
        frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise HouseDataError("请上传 UTF-8 编码的有效 CSV 文件") from exc
    if TARGET_COLUMN not in frame.columns:
        raise HouseDataError(f"CSV 必须包含目标列 {TARGET_COLUMN}")
    if frame.columns.duplicated().any():
        raise HouseDataError("CSV 列名不能重复")
    if not 100 <= len(frame) <= 5000:
        raise HouseDataError("数据需要 100–5000 行")
    if len(frame.columns) > 100:
        raise HouseDataError("CSV 列数不能超过 100 列")
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy()).all():
        raise HouseDataError(f"{TARGET_COLUMN} 必须全部是有效数字")
    if (target <= 0).any():
        raise HouseDataError(f"{TARGET_COLUMN} 必须全部大于 0")

    features = frame.drop(columns=[TARGET_COLUMN, "Id"], errors="ignore")
    if features.empty:
        raise HouseDataError("请至少保留一个特征列")
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    for column in categorical:
        # 统一成字符串，避免 1 与 "1" 被 OneHot 当成两个取值
        features[column] = features[column].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        ).astype(object)
    return features, target.astype(float), numeric, categorical


def evaluate_regression(
    model: nn.Module,
    features: torch.Tensor,
    normalized_log_y: torch.Tensor,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        predicted_normalized = model(features).squeeze(1)
        predicted_log = predicted_normalized * target_std + target_mean
        actual_log = normalized_log_y * target_std + target_mean
        log_rmse = torch.sqrt(torch.mean((predicted_log - actual_log) ** 2)).item()
        predicted_price = torch.exp(predicted_log.clamp(max=20))
        actual_price = torch.exp(actual_log)
        mae = torch.mean(torch.abs(predicted_price - actual_price)).item()
        denominator = torch.sum((actual_price - actual_price.mean()) ** 2)
        r2 = (
            1 - torch.sum((predicted_price - actual_price) ** 2) / denominator
        ).item() if denominator > 0 else 0.0
    return {
        "log_rmse": round(log_rmse, 5),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
    }


class HouseService:
    """房价模型的训练、持久化与推理。"""

    algorithm = "mlp"

    def __init__(self, root: Path) -> None:
        self.store = ModelStore(root, (WEIGHT_FILE, PREPROCESSOR_FILE), "房价模型")
        self.jobs = JobRegistry("house-train", "已有房价模型正在训练，请等待当前任务结束后再提交")
        self._lock = self.jobs.lock
        self._cached_id: str | None = None
        self._cached_model: nn.Module | None = None
        self._cached_preprocessor: ColumnTransformer | None = None
        self._cached_metadata: dict[str, Any] | None = None

    # ---------- 查询 ----------

    def list_models(self) -> list[dict[str, Any]]:
        return self.store.list_models()

    def job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    # ---------- 训练 ----------

    def submit(self, raw: bytes, name: str, epochs: int) -> dict[str, Any]:
        name = name.strip()
        if not 1 <= len(name) <= 80:
            raise HouseDataError("模型名称需为 1–80 个字符")
        if not 1 <= epochs <= 200:
            raise HouseDataError("训练轮数需在 1–200 之间")
        features, targets, numeric, categorical = read_dataset(raw)
        fingerprint = sha256_hex(raw)
        return self.jobs.submit(
            epochs=epochs,
            worker=lambda job_id: self._train(
                job_id, features, targets, numeric, categorical, name, epochs, fingerprint
            ),
            row_count=int(len(features)),
            input_dim=len(features.columns),
            numeric_count=len(numeric),
            categorical_count=len(categorical),
        )

    def _train(
        self,
        job_id: str,
        features: pd.DataFrame,
        targets: pd.Series,
        numeric: list[str],
        categorical: list[str],
        name: str,
        epochs: int,
        fingerprint: str,
    ) -> None:
        staging = self.store.staging(job_id)
        started = time.perf_counter()
        try:
            torch.manual_seed(SEED)
            torch.set_num_threads(4)
            self.jobs.update(job_id, status="training", stage="划分数据并拟合预处理器")

            indices = np.arange(len(features))
            train_val_idx, test_idx = train_test_split(
                indices, test_size=0.2, random_state=SEED, shuffle=True
            )
            train_idx, validation_idx = train_test_split(
                train_val_idx, test_size=0.2, random_state=SEED, shuffle=True
            )
            train_frame = features.iloc[train_idx]
            preprocessor = make_preprocessor(numeric, categorical)
            train_x = np.asarray(preprocessor.fit_transform(train_frame), dtype=np.float32)
            if train_x.shape[1] > MAX_FEATURE_DIM:
                raise HouseDataError(f"独热编码后特征维度过大（{train_x.shape[1]} > {MAX_FEATURE_DIM}）")
            if not np.isfinite(train_x).all():
                raise HouseDataError("特征中存在无法处理的数值")
            validation_x = np.asarray(
                preprocessor.transform(features.iloc[validation_idx]), dtype=np.float32
            )
            test_x = np.asarray(preprocessor.transform(features.iloc[test_idx]), dtype=np.float32)

            train_log = np.log(targets.iloc[train_idx].to_numpy(dtype=np.float64))
            target_mean = float(train_log.mean())
            target_std = max(float(train_log.std()), 0.05)
            train_y = torch.tensor((train_log - target_mean) / target_std, dtype=torch.float32)
            validation_y = torch.tensor(
                (np.log(targets.iloc[validation_idx].to_numpy(dtype=np.float64)) - target_mean)
                / target_std,
                dtype=torch.float32,
            )
            test_y = torch.tensor(
                (np.log(targets.iloc[test_idx].to_numpy(dtype=np.float64)) - target_mean)
                / target_std,
                dtype=torch.float32,
            )

            batch_size = 63 if len(train_idx) % 64 == 1 else 64
            loader = DataLoader(
                TensorDataset(torch.from_numpy(train_x), train_y),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(SEED),
            )

            model = make_model(train_x.shape[1])
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_function = nn.MSELoss()

            best_loss, best_epoch, best_state = float("inf"), 0, None
            validation_tensor = torch.from_numpy(validation_x)
            for epoch in range(1, epochs + 1):
                model.train()
                squared_error = 0.0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    loss = loss_function(model(batch_x).squeeze(1), batch_y)
                    loss.backward()
                    optimizer.step()
                    squared_error += loss.item() * len(batch_y)
                train_log_rmse = math.sqrt(squared_error / len(train_idx)) * target_std
                validation = evaluate_regression(
                    model, validation_tensor, validation_y, target_mean, target_std
                )
                if validation["log_rmse"] < best_loss:
                    best_loss, best_epoch = validation["log_rmse"], epoch
                    best_state = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                self.jobs.record_epoch(
                    job_id,
                    {
                        "epoch": epoch,
                        "train_log_rmse": round(train_log_rmse, 5),
                        "validation_log_rmse": validation["log_rmse"],
                        "validation_mae": validation["mae"],
                    },
                    epoch,
                    epochs,
                    f"第 {epoch}/{epochs} 轮，验证 Log RMSE {validation['log_rmse']:.4f}",
                )

            if best_state is None:
                raise HouseDataError("训练未能产生有效权重")
            model.load_state_dict(best_state)
            test = evaluate_regression(model, torch.from_numpy(test_x), test_y, target_mean, target_std)

            self.jobs.update(job_id, status="saving", stage="保存最佳权重、预处理器与测试结果")
            defaults: dict[str, Any] = {}
            categories: dict[str, list[str]] = {}
            for column in numeric:
                series = pd.to_numeric(train_frame[column], errors="coerce")
                defaults[column] = float(series.median()) if series.notna().any() else 0.0
            for column in categorical:
                values = train_frame[column].dropna().astype(str)
                defaults[column] = values.mode().iloc[0] if not values.empty else "NaN"
                categories[column] = values.value_counts().head(50).index.tolist()

            staging.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), staging / WEIGHT_FILE)
            joblib.dump(preprocessor, staging / PREPROCESSOR_FILE)
            metadata = {
                "id": job_id,
                "name": name,
                "algorithm": self.algorithm,
                "algorithm_label": "MLP 回归（128 隐层 + BatchNorm + Dropout 0.2）",
                "created_at": utc_now(),
                "data_fingerprint": fingerprint,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "history": self.jobs.get(job_id)["history"],
                "test": test,
                "train_seconds": round(time.perf_counter() - started, 2),
                "model_bytes": file_tree_bytes(staging),
                "param_count": sum(parameter.numel() for parameter in model.parameters()),
                "row_count": int(len(features)),
                "train_count": int(len(train_idx)),
                "validation_count": int(len(validation_idx)),
                "test_count": int(len(test_idx)),
                "feature_columns": features.columns.tolist(),
                "numeric_columns": numeric,
                "categorical_columns": categorical,
                "input_dim": int(train_x.shape[1]),
                "target_column": TARGET_COLUMN,
                "target_mean": target_mean,
                "target_std": target_std,
                "defaults": defaults,
                "categories": categories,
            }
            self.store.commit(staging, job_id, metadata)
            self.jobs.update(
                job_id, status="completed", stage="训练完成，模型已保存", model=metadata
            )
        finally:
            self.store.discard(staging)

    # ---------- 推理 ----------

    def _load(self, model_id: str) -> None:
        with self._lock:
            folder = self.store.folder(model_id)
            if self._cached_id == model_id:
                return
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            model = make_model(int(metadata["input_dim"]))
            model.load_state_dict(
                torch.load(folder / WEIGHT_FILE, map_location="cpu", weights_only=True)
            )
            model.eval()
            self._cached_id = model_id
            self._cached_model = model
            self._cached_preprocessor = joblib.load(folder / PREPROCESSOR_FILE)
            self._cached_metadata = metadata

    def predict(self, model_id: str, features: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._load(model_id)
            assert self._cached_metadata is not None
            assert self._cached_model is not None
            assert self._cached_preprocessor is not None
            metadata = self._cached_metadata
            known = set(metadata["feature_columns"])
            if not isinstance(features, dict) or set(features) - known:
                raise HouseDataError("提交的房屋特征包含未知字段")

            record = dict(metadata["defaults"])
            for column, value in features.items():
                if column in metadata["numeric_columns"]:
                    if value in (None, ""):
                        record[column] = np.nan
                        continue
                    try:
                        number = float(value)
                    except (TypeError, ValueError) as exc:
                        raise HouseDataError(f"{column} 必须是数字") from exc
                    if not math.isfinite(number) or abs(number) > 1e12:
                        raise HouseDataError(f"{column} 的数值超出有效范围")
                    record[column] = number
                else:
                    if value is not None and (not isinstance(value, str) or len(value) > 100):
                        raise HouseDataError(f"{column} 的类别值无效（需为不超过 100 字的文本）")
                    record[column] = value if value else np.nan

            row = pd.DataFrame([record], columns=metadata["feature_columns"])
            if metadata["numeric_columns"]:
                row[metadata["numeric_columns"]] = row[metadata["numeric_columns"]].apply(
                    pd.to_numeric, errors="coerce"
                )
            transformed = np.asarray(
                self._cached_preprocessor.transform(row), dtype=np.float32
            )
            with torch.no_grad():
                normalized = float(self._cached_model(torch.from_numpy(transformed)).item())
            log_price = normalized * metadata["target_std"] + metadata["target_mean"]
            price = math.exp(min(log_price, 20))
            return {
                "data": {"price": round(price, 2), "currency": "USD"},
                "model_id": model_id,
                "notice": "未填写的特征使用训练集的中位数/众数填充；结果为点估计，不含置信区间，不构成投资建议。",
            }

    # ---------- 删除 ----------

    def delete(self, model_id: str) -> None:
        with self._lock:
            folder = self.store.folder(model_id)
            if self._cached_id == model_id:
                self._cached_id = None
                self._cached_model = None
                self._cached_preprocessor = None
                self._cached_metadata = None
            self.store.delete(folder.name)


houses = HouseService(Path(os.getenv(
    "HOUSE_MODEL_DIR",
    str(Path(__file__).resolve().parents[2] / "artifacts" / "house"),
)))
