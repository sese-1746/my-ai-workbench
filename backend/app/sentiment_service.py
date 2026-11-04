"""模块B 情感分析：中文文本分类（双向 GRU / TextCNN 可切换）。

数据 → jieba 分词 → 词表（≤12000）→ 截断 128 token → 分层 70/15/15 划分；
按验证集 loss 保存最佳权重，训练结束后一次性评估留出测试集并给出混淆矩阵。
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import jieba
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from .common import (
    SEED,
    DataError,
    JobRegistry,
    ModelStore,
    file_tree_bytes,
    sha256_hex,
    utc_now,
)

MAX_BYTES = 5 * 1024 * 1024
MIN_ROWS, MAX_ROWS = 50, 5000
MIN_PER_CLASS = 5
MIN_CLASSES, MAX_CLASSES = 2, 12
MAX_VOCAB = 12000
MAX_LEN = 128
MAX_TEXT_CHARS = 2000
MAX_LABEL_CHARS = 50
BATCH_SIZE = 32
WEIGHT_FILE = "weights.pt"
VOCAB_FILE = "vocab.json"

PAD, UNK = 0, 1
ALGORITHMS = {"bigru": "双向 GRU（64 维）", "textcnn": "TextCNN（卷积核 3/4/5）"}

TEXT_KEYS = ("text", "content", "正文", "文本", "评论", "内容")
TITLE_KEYS = ("title", "标题")
LABEL_KEYS = ("label", "类别", "情感", "标签")

LATIN_TOKEN = re.compile(r"[a-zA-Z0-9']+")
PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)


class SentimentDataError(DataError):
    """情感数据或输入文本不合法。"""


class BiGruClassifier(nn.Module):
    def __init__(self, vocab_size: int, class_count: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(embedding_dim, 64, batch_first=True, bidirectional=True)
        self.head = nn.Linear(128, class_count)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        # hidden: (2, batch, 64) → 拼接正反向末状态
        features = torch.cat([hidden[0], hidden[1]], dim=1)
        return self.head(self.dropout(features))


class TextCnnClassifier(nn.Module):
    def __init__(self, vocab_size: int, class_count: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(0.3)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(embedding_dim, 64, kernel_size=size) for size in (3, 4, 5)]
        )
        self.head = nn.Linear(64 * 3, class_count)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(tokens)).transpose(1, 2)  # (B, E, L)
        mask = (
            torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
        ).unsqueeze(1)  # (B, 1, L)
        pooled = []
        for convolution in self.convolutions:
            activated = torch.relu(convolution(embedded))
            window_mask = mask[:, :, : activated.shape[2]]
            # 用极小值填补 padding 位置，保证 max 只取有效 token
            activated = activated.masked_fill(~window_mask, torch.finfo(activated.dtype).min)
            pooled.append(activated.max(dim=2).values)
        return self.head(self.dropout(torch.cat(pooled, dim=1)))


def build_model(algorithm: str, vocab_size: int, class_count: int) -> nn.Module:
    if algorithm == "bigru":
        return BiGruClassifier(vocab_size, class_count)
    if algorithm == "textcnn":
        return TextCnnClassifier(vocab_size, class_count)
    raise SentimentDataError("算法只能是 bigru 或 textcnn")


def tokenize(text: str, tokenizer: str = "jieba") -> list[str]:
    """中文优先用 jieba，英文片段用正则兜底，过滤纯标点。"""
    lowered = text.lower()
    tokens: list[str] = []
    if tokenizer == "jieba":
        for piece in jieba.lcut(lowered):
            piece = piece.strip()
            if not piece or PUNCT_ONLY.match(piece):
                continue
            if LATIN_TOKEN.fullmatch(piece) or re.search(r"[\u4e00-\u9fff]", piece):
                tokens.append(piece)
    else:
        tokens = LATIN_TOKEN.findall(lowered)
    return tokens


def parse_dataset(raw: bytes, file_type: str) -> tuple[list[str], list[str]]:
    if not raw:
        raise SentimentDataError("数据文件为空，请重新选择")
    if len(raw) > MAX_BYTES:
        raise SentimentDataError("数据文件不能超过 5 MB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SentimentDataError("文件需使用 UTF-8 编码") from exc

    records: list[tuple[str, str]] = []
    if file_type == "jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SentimentDataError(f"第 {line_no} 行不是合法 JSON") from exc
            if not isinstance(item, dict):
                raise SentimentDataError(f"第 {line_no} 行必须是 JSON 对象")
            body = _pick(item, TEXT_KEYS)
            title = _pick(item, TITLE_KEYS)
            label = _pick(item, LABEL_KEYS)
            records.append((f"{title} {body}".strip(), str(label).strip()))
    else:
        try:
            frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            raise SentimentDataError("请上传 UTF-8 编码、带表头的 CSV") from exc
        if frame.empty:
            raise SentimentDataError("CSV 没有数据行")
        text_column = _find_column(frame.columns, TEXT_KEYS)
        label_column = _find_column(frame.columns, LABEL_KEYS)
        if text_column is None or label_column is None:
            raise SentimentDataError("CSV 需要 text/label 两列（也接受 正文/类别）")
        title_column = _find_column(frame.columns, TITLE_KEYS)
        for _, row in frame.iterrows():
            body = "" if pd.isna(row[text_column]) else str(row[text_column])
            title = "" if title_column is None or pd.isna(row[title_column]) else str(row[title_column])
            label = "" if pd.isna(row[label_column]) else str(row[label_column])
            records.append((f"{title} {body}".strip(), label.strip()))

    cleaned: list[tuple[str, str]] = []
    for body, label in records:
        if not body or not label:
            raise SentimentDataError("每条样本都必须有文本和类别标签")
        if len(body) > MAX_TEXT_CHARS:
            raise SentimentDataError(f"单条文本不能超过 {MAX_TEXT_CHARS} 字")
        if len(label) > MAX_LABEL_CHARS:
            raise SentimentDataError(f"类别名不能超过 {MAX_LABEL_CHARS} 字")
        cleaned.append((body, label))

    if not MIN_ROWS <= len(cleaned) <= MAX_ROWS:
        raise SentimentDataError(f"样本数需在 {MIN_ROWS}–{MAX_ROWS} 条之间，当前 {len(cleaned)} 条")
    return cleaned


def validate_classes(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if not MIN_CLASSES <= len(counts) <= MAX_CLASSES:
        raise SentimentDataError(f"类别数需在 {MIN_CLASSES}–{MAX_CLASSES} 之间，当前 {len(counts)} 个")
    sparse = [f"{label}（{count} 条）" for label, count in counts.items() if count < MIN_PER_CLASS]
    if sparse:
        raise SentimentDataError(f"每个类别至少 {MIN_PER_CLASS} 条，不足的有：" + "、".join(sparse))
    return sorted(counts)


def _pick(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _find_column(columns, keys: tuple[str, ...]) -> str | None:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


class EncodedDataset(Dataset):
    def __init__(self, sequences: list[list[int]], labels: list[int]) -> None:
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = torch.tensor(self.sequences[index], dtype=torch.long)
        return ids, torch.tensor(len(ids), dtype=torch.long), torch.tensor(self.labels[index])


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    sequences, lengths, labels = zip(*batch)
    # TextCNN 的卷积核最大为 5，短句也补齐到 5，避免 Conv1d 报错
    max_length = max(int(max(lengths).item()), 5)
    padded = torch.zeros(len(batch), max_length, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = sequence
    return padded, torch.stack(lengths), torch.stack(labels)


def encode(text: str, vocab: dict[str, int], tokenizer: str) -> list[int]:
    tokens = tokenize(text, tokenizer)
    if not tokens:
        raise SentimentDataError("文本中没有可用于分析的有效字符")
    ids = [vocab.get(token, UNK) for token in tokens[:MAX_LEN]]
    return ids or [UNK]


class SentimentService:
    """情感模型的训练、持久化与推理。"""

    def __init__(self, root: Path) -> None:
        self.store = ModelStore(root, (WEIGHT_FILE, VOCAB_FILE), "情感模型")
        self.jobs = JobRegistry("sentiment-train", "已有情感模型正在训练，请等待当前任务结束后再提交")
        self._lock = self.jobs.lock
        self._cached_id: str | None = None
        self._cached_model: nn.Module | None = None
        self._cached_vocab: dict[str, int] | None = None
        self._cached_labels: list[str] | None = None
        self._cached_tokenizer = "jieba"
        self._cached_metadata: dict[str, Any] | None = None

    # ---------- 查询 ----------

    def list_models(self) -> list[dict[str, Any]]:
        return self.store.list_models()

    def job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    # ---------- 训练 ----------

    def submit(
        self, raw: bytes, file_type: str, name: str, algorithm: str, tokenizer: str, epochs: int
    ) -> dict[str, Any]:
        name = name.strip()
        if not 1 <= len(name) <= 80:
            raise SentimentDataError("模型名称需为 1–80 个字符")
        if algorithm not in ALGORITHMS:
            raise SentimentDataError("算法只能是 bigru 或 textcnn")
        if tokenizer not in ("jieba", "regex"):
            raise SentimentDataError("分词器只能是 jieba 或 regex")
        if not 1 <= epochs <= 20:
            raise SentimentDataError("训练轮数需在 1–20 之间")
        samples = parse_dataset(raw, file_type)
        texts = [item[0] for item in samples]
        labels = [item[1] for item in samples]
        class_names = validate_classes(labels)
        return self.jobs.submit(
            epochs=epochs,
            worker=lambda job_id: self._train(
                job_id, texts, labels, class_names, name, algorithm, tokenizer, epochs,
                sha256_hex(raw),
            ),
            row_count=len(texts),
            class_count=len(class_names),
            algorithm=algorithm,
        )

    def _train(
        self,
        job_id: str,
        texts: list[str],
        labels: list[str],
        class_names: list[str],
        name: str,
        algorithm: str,
        tokenizer: str,
        epochs: int,
        fingerprint: str,
    ) -> None:
        staging = self.store.staging(job_id)
        started = time.perf_counter()
        try:
            torch.manual_seed(SEED)
            torch.set_num_threads(4)
            self.jobs.update(job_id, status="training", stage="分层划分数据并构建词表")

            label_to_index = {label: index for index, label in enumerate(class_names)}
            encoded_labels = [label_to_index[label] for label in labels]
            indices = np.arange(len(texts))
            train_val_idx, test_idx = train_test_split(
                indices, test_size=0.15, random_state=SEED, stratify=encoded_labels
            )
            train_idx, validation_idx = train_test_split(
                train_val_idx, test_size=0.1765, random_state=SEED,
                stratify=[encoded_labels[i] for i in train_val_idx],
            )

            tokenized_train = [tokenize(texts[i], tokenizer)[:MAX_LEN] for i in train_idx]
            frequency: dict[str, int] = {}
            for tokens in tokenized_train:
                for token in tokens:
                    frequency[token] = frequency.get(token, 0) + 1
            ordered = sorted(frequency.items(), key=lambda pair: (-pair[1], pair[0]))[: MAX_VOCAB - 2]
            vocab = {"<pad>": PAD, "<unk>": UNK}
            for token, _ in ordered:
                vocab[token] = len(vocab)
            if len(vocab) <= 2:
                raise SentimentDataError("训练集分词后没有有效词，请检查文本内容")

            def encode_indices(rows) -> tuple[list[list[int]], list[int]]:
                sequences, targets = [], []
                for row in rows:
                    ids = [vocab.get(token, UNK) for token in tokenize(texts[row], tokenizer)[:MAX_LEN]]
                    sequences.append(ids or [UNK])
                    targets.append(encoded_labels[row])
                return sequences, targets

            train_sequences, train_targets = encode_indices(train_idx)
            validation_sequences, validation_targets = encode_indices(validation_idx)
            test_sequences, test_targets = encode_indices(test_idx)

            generator = torch.Generator().manual_seed(SEED)
            train_loader = DataLoader(
                EncodedDataset(train_sequences, train_targets),
                batch_size=BATCH_SIZE, shuffle=True, generator=generator, collate_fn=collate,
            )
            validation_loader = DataLoader(
                EncodedDataset(validation_sequences, validation_targets),
                batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
            )
            test_loader = DataLoader(
                EncodedDataset(test_sequences, test_targets),
                batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
            )

            model = build_model(algorithm, len(vocab), len(class_names))
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
            loss_function = nn.CrossEntropyLoss()
            best_loss, best_epoch, best_state = float("inf"), 0, None

            for epoch in range(1, epochs + 1):
                model.train()
                total_loss, correct, seen = 0.0, 0, 0
                for batch_tokens, lengths, batch_labels in train_loader:
                    optimizer.zero_grad()
                    logits = model(batch_tokens, lengths)
                    loss = loss_function(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_labels)
                    correct += int((logits.argmax(dim=1) == batch_labels).sum())
                    seen += len(batch_labels)

                validation = self._evaluate(model, validation_loader)
                entry = {
                    "epoch": epoch,
                    "train_loss": round(total_loss / max(seen, 1), 5),
                    "train_accuracy": round(correct / max(seen, 1), 4),
                    "validation_loss": validation["loss"],
                    "validation_accuracy": validation["accuracy"],
                    "validation_macro_f1": validation["macro_f1"],
                }
                if validation["loss"] < best_loss:
                    best_loss, best_epoch = validation["loss"], epoch
                    best_state = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                self.jobs.record_epoch(
                    job_id, entry, epoch, epochs,
                    f"第 {epoch}/{epochs} 轮，验证准确率 {validation['accuracy']:.2%}",
                )

            if best_state is None:
                raise SentimentDataError("训练未能产生有效权重")
            model.load_state_dict(best_state)
            test = self._evaluate(model, test_loader, with_matrix=True, class_count=len(class_names))

            self.jobs.update(job_id, status="saving", stage="保存最佳权重、词表与测试结果")
            staging.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), staging / WEIGHT_FILE)
            (staging / VOCAB_FILE).write_text(
                json.dumps({"tokens": list(vocab.keys()), "tokenizer": tokenizer}, ensure_ascii=False),
                encoding="utf-8",
            )
            metadata = {
                "id": job_id,
                "name": name,
                "algorithm": algorithm,
                "algorithm_label": ALGORITHMS[algorithm],
                "tokenizer": tokenizer,
                "created_at": utc_now(),
                "data_fingerprint": fingerprint,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "history": self.jobs.get(job_id)["history"],
                "test": test,
                "labels": class_names,
                "vocab_size": len(vocab),
                "max_length": MAX_LEN,
                "train_seconds": round(time.perf_counter() - started, 2),
                "model_bytes": file_tree_bytes(staging),
                "param_count": sum(parameter.numel() for parameter in model.parameters()),
                "row_count": len(texts),
                "train_count": len(train_idx),
                "validation_count": len(validation_idx),
                "test_count": len(test_idx),
            }
            self.store.commit(staging, job_id, metadata)
            self.jobs.update(
                job_id, status="completed", stage="训练完成，模型已保存", model=metadata
            )
        finally:
            self.store.discard(staging)

    @staticmethod
    def _evaluate(
        model: nn.Module,
        loader: DataLoader,
        with_matrix: bool = False,
        class_count: int | None = None,
    ) -> dict[str, Any]:
        model.eval()
        total_loss, predictions, targets = 0.0, [], []
        loss_function = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_tokens, lengths, batch_labels in loader:
                logits = model(batch_tokens, lengths)
                total_loss += loss_function(logits, batch_labels).item() * len(batch_labels)
                predictions.extend(logits.argmax(dim=1).tolist())
                targets.extend(batch_labels.tolist())
        result: dict[str, Any] = {
            "loss": round(total_loss / max(len(targets), 1), 5),
            "accuracy": round(accuracy_score(targets, predictions), 4) if targets else 0.0,
            "macro_f1": round(f1_score(targets, predictions, average="macro", zero_division=0), 4)
            if targets else 0.0,
        }
        if with_matrix:
            labels = list(range(class_count or 0))
            matrix = confusion_matrix(targets, predictions, labels=labels)
            result["confusion_matrix"] = matrix.tolist()
        return result

    # ---------- 推理 ----------

    def _load(self, model_id: str) -> None:
        with self._lock:
            folder = self.store.folder(model_id)
            if self._cached_id == model_id:
                return
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            vocab_payload = json.loads((folder / VOCAB_FILE).read_text(encoding="utf-8"))
            vocab = {token: index for index, token in enumerate(vocab_payload["tokens"])}
            model = build_model(metadata["algorithm"], len(vocab), len(metadata["labels"]))
            model.load_state_dict(
                torch.load(folder / WEIGHT_FILE, map_location="cpu", weights_only=True)
            )
            model.eval()
            self._cached_id = model_id
            self._cached_model = model
            self._cached_vocab = vocab
            self._cached_labels = metadata["labels"]
            self._cached_tokenizer = metadata.get("tokenizer", "jieba")
            self._cached_metadata = metadata

    def classify(self, model_id: str, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise SentimentDataError("请输入需要分析的中文文本")
        if len(text) > MAX_TEXT_CHARS:
            raise SentimentDataError(f"文本不能超过 {MAX_TEXT_CHARS} 字")
        with self._lock:
            self._load(model_id)
            assert self._cached_model is not None and self._cached_vocab is not None
            assert self._cached_labels is not None and self._cached_metadata is not None
            tokens = tokenize(text, self._cached_tokenizer)
            ids = [self._cached_vocab.get(token, UNK) for token in tokens[:MAX_LEN]] or [UNK]
            real_length = len(ids)
            if len(ids) < 5:  # TextCNN 卷积核最大为 5，短文本补齐
                ids = ids + [PAD] * (5 - len(ids))
            token_tensor = torch.tensor([ids], dtype=torch.long)
            length_tensor = torch.tensor([real_length], dtype=torch.long)
            with torch.no_grad():
                probabilities = torch.softmax(self._cached_model(token_tensor, length_tensor), dim=1)[0]
            ranked = sorted(
                (
                    {"label": label, "probability": round(float(probabilities[index]), 4)}
                    for index, label in enumerate(self._cached_labels)
                ),
                key=lambda item: item["probability"],
                reverse=True,
            )
            return {
                "data": {
                    "label": ranked[0]["label"],
                    "confidence": ranked[0]["probability"],
                    "probabilities": ranked,
                    "tokens": tokens[:24],
                    "token_count": len(tokens),
                },
                "model_id": model_id,
                "notice": "类别概率未经校准，仅用于演示，不能作为业务决策的唯一依据。",
            }

    # ---------- 删除 ----------

    def delete(self, model_id: str) -> None:
        with self._lock:
            self.store.folder(model_id)
            if self._cached_id == model_id:
                self._cached_id = None
                self._cached_model = None
                self._cached_vocab = None
                self._cached_labels = None
                self._cached_metadata = None
            self.store.delete(model_id)


sentiments = SentimentService(Path(os.getenv(
    "SENTIMENT_MODEL_DIR",
    str(Path(__file__).resolve().parents[2] / "artifacts" / "sentiment"),
)))
