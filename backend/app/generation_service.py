"""模块C 文本生成：字符级 Decoder-only Transformer。

语料 → 字符词表（≤3000）→ 上下文 96 字符滑窗 → 2 层 4 头因果注意力；
按验证集 loss 保存最佳权重，推理时用 top-k + temperature 采样续写。
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from .common import (
    SEED,
    DataError,
    JobRegistry,
    ModelStore,
    file_tree_bytes,
    sha256_hex,
    utc_now,
)

MAX_BYTES = 2 * 1024 * 1024
MIN_DOCS, MAX_DOCS = 10, 2000
MAX_TOTAL_CHARS = 120_000
MAX_VOCAB = 3000
CONTEXT = 96
WIDTH, LAYERS, HEADS = 96, 2, 4
BATCH_SIZE = 32
WEIGHT_FILE = "weights.pt"
VOCAB_FILE = "vocab.json"

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")


class GenerationDataError(DataError):
    """生成语料或生成参数不合法。"""


class CausalDecoder(nn.Module):
    """Pre-LN 的 Decoder-only Transformer（无 cross-attention）。"""

    def __init__(
        self,
        vocab_size: int,
        width: int = WIDTH,
        layers: int = LAYERS,
        heads: int = HEADS,
        context: int = CONTEXT,
    ) -> None:
        super().__init__()
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, width, padding_idx=PAD)
        self.position_embedding = nn.Embedding(context, width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        hidden = self.blocks(hidden, mask=causal_mask)
        return self.head(self.norm(hidden))


def parse_corpus(raw: bytes, file_type: str) -> list[str]:
    """解析语料并切成「一行 = 一段文本」的列表。"""
    if not raw:
        raise GenerationDataError("语料文件为空，请重新选择")
    if len(raw) > MAX_BYTES:
        raise GenerationDataError("语料文件不能超过 2 MB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GenerationDataError("文件需使用 UTF-8 编码") from exc
    if not text.strip():
        raise GenerationDataError("语料内容为空")

    documents: list[str] = []
    if file_type == "csv":
        try:
            frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            raise GenerationDataError("请上传 UTF-8 编码、带表头的 CSV") from exc
        column = _find_column(frame.columns, ("content", "text", "正文", "文本"))
        if column is None:
            raise GenerationDataError("CSV 需要 content 或 正文 列")
        title_column = _find_column(frame.columns, ("title", "标题"))
        for _, row in frame.iterrows():
            body = "" if pd.isna(row[column]) else str(row[column]).strip()
            title = "" if title_column is None or pd.isna(row[title_column]) else str(row[title_column]).strip()
            line = f"{title} {body}".strip() if title else body
            if line:
                documents.extend(part.strip() for part in line.splitlines() if part.strip())
    else:
        documents = [line.strip() for line in text.splitlines() if line.strip()]

    if len(documents) < MIN_DOCS:
        # 行数不足时按句子再切一次，避免短语料被直接拒绝
        documents = [part.strip() for part in SENTENCE_SPLIT.split(text) if part.strip()]

    # 去重但保持原顺序
    unique: list[str] = []
    seen: set[str] = set()
    for document in documents:
        if document not in seen:
            seen.add(document)
            unique.append(document)

    if len(unique) < MIN_DOCS:
        raise GenerationDataError(f"去重后至少需要 {MIN_DOCS} 条语料，当前 {len(unique)} 条")
    if len(unique) > MAX_DOCS:
        raise GenerationDataError(f"语料条数不能超过 {MAX_DOCS} 条")
    total = sum(len(document) for document in unique)
    if total > MAX_TOTAL_CHARS:
        raise GenerationDataError(f"语料总长度不能超过 {MAX_TOTAL_CHARS} 字符，当前 {total} 字符")
    if total < 200:
        raise GenerationDataError("语料过短（至少 200 字符），无法训练语言模型")
    return unique


def _find_column(columns, keys: tuple[str, ...]) -> str | None:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for key in keys:
        if key in lowered:
            return lowered[key]
    return None


def build_samples(ids: list[int]) -> list[tuple[list[int], list[int]]]:
    """按 95 步长滑窗构造 (输入, 目标)，长度均不超过 CONTEXT-1。"""
    samples: list[tuple[list[int], list[int]]] = []
    step = CONTEXT - 1
    for start in range(0, max(len(ids) - 1, 1), step):
        chunk = ids[start:start + CONTEXT]
        if len(chunk) < 2:
            break
        samples.append((chunk[:-1], chunk[1:]))
    return samples


def pad_batch(samples: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(source) for source, _ in samples)
    source_batch = torch.full((len(samples), width), PAD, dtype=torch.long)
    target_batch = torch.full((len(samples), width), PAD, dtype=torch.long)
    for row, (source, target) in enumerate(samples):
        source_batch[row, : len(source)] = torch.tensor(source, dtype=torch.long)
        target_batch[row, : len(target)] = torch.tensor(target, dtype=torch.long)
    return source_batch, target_batch


class GenerationService:
    """字符级语言模型的训练、持久化与采样生成。"""

    def __init__(self, root: Path) -> None:
        self.store = ModelStore(root, (WEIGHT_FILE, VOCAB_FILE), "生成模型")
        self.jobs = JobRegistry("generation-train", "已有生成模型正在训练，请等待当前任务结束后再提交")
        self._lock = self.jobs.lock
        self._cached_id: str | None = None
        self._cached_model: CausalDecoder | None = None
        self._cached_vocab: list[str] | None = None
        self._cached_index: dict[str, int] | None = None
        self._cached_metadata: dict[str, Any] | None = None

    # ---------- 查询 ----------

    def list_models(self) -> list[dict[str, Any]]:
        return self.store.list_models()

    def job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    # ---------- 训练 ----------

    def submit(self, raw: bytes, file_type: str, name: str, epochs: int) -> dict[str, Any]:
        name = name.strip()
        if not 1 <= len(name) <= 80:
            raise GenerationDataError("模型名称需为 1–80 个字符")
        if not 1 <= epochs <= 20:
            raise GenerationDataError("训练轮数需在 1–20 之间")
        documents = parse_corpus(raw, file_type)
        return self.jobs.submit(
            epochs=epochs,
            worker=lambda job_id: self._train(
                job_id, documents, name, epochs, sha256_hex(raw)
            ),
            document_count=len(documents),
            char_count=sum(len(document) for document in documents),
        )

    def _train(
        self,
        job_id: str,
        documents: list[str],
        name: str,
        epochs: int,
        fingerprint: str,
    ) -> None:
        staging = self.store.staging(job_id)
        started = time.perf_counter()
        try:
            torch.manual_seed(SEED)
            torch.set_num_threads(4)
            self.jobs.update(job_id, status="training", stage="划分语料并构建字符词表")

            train_docs, rest_docs = train_test_split(
                documents, test_size=0.2, random_state=SEED, shuffle=True
            )
            validation_docs, test_docs = train_test_split(
                rest_docs, test_size=0.5, random_state=SEED, shuffle=True
            )

            frequency: dict[str, int] = {}
            for document in train_docs:
                for character in document:
                    frequency[character] = frequency.get(character, 0) + 1
            ordered = sorted(frequency.items(), key=lambda pair: (-pair[1], pair[0]))[: MAX_VOCAB - 4]
            if len(ordered) < 10:
                raise GenerationDataError("语料字符种类过少，无法训练语言模型")
            vocab = list(SPECIAL_TOKENS) + [character for character, _ in ordered]
            token_to_id = {character: index for index, character in enumerate(vocab)}

            def encode(text: str) -> list[int]:
                return [token_to_id.get(character, UNK) for character in text]

            train_samples: list[tuple[list[int], list[int]]] = []
            for document in train_docs:
                train_samples.extend(build_samples(encode(document)))
            if len(train_samples) < BATCH_SIZE:
                raise GenerationDataError("训练样本过少，请提供更长的语料")
            validation_ids = [BOS] + encode("".join(validation_docs)) + [EOS]
            test_ids = [BOS] + encode("".join(test_docs)) + [EOS]

            model = CausalDecoder(len(vocab))
            for parameter in model.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss_function = nn.CrossEntropyLoss(ignore_index=PAD)
            generator = torch.Generator().manual_seed(SEED)

            best_loss, best_epoch, best_state = float("inf"), 0, None
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_samples), generator=generator).tolist()
                total_loss, batches = 0.0, 0
                for start in range(0, len(order), BATCH_SIZE):
                    batch = [train_samples[index] for index in order[start:start + BATCH_SIZE]]
                    source, target = pad_batch(batch)
                    optimizer.zero_grad()
                    logits = model(source)
                    loss = loss_function(logits.reshape(-1, len(vocab)), target.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    batches += 1

                validation_loss = self._evaluate_loss(model, validation_ids, loss_function, len(vocab))
                entry = {
                    "epoch": epoch,
                    "train_loss": round(total_loss / max(batches, 1), 4),
                    "validation_loss": round(validation_loss, 4),
                    "validation_perplexity": round(math.exp(min(validation_loss, 20)), 2),
                }
                if validation_loss < best_loss:
                    best_loss, best_epoch = validation_loss, epoch
                    best_state = {
                        key: value.detach().clone() for key, value in model.state_dict().items()
                    }
                self.jobs.record_epoch(
                    job_id, entry, epoch, epochs,
                    f"第 {epoch}/{epochs} 轮，验证困惑度 {entry['validation_perplexity']}",
                )

            if best_state is None:
                raise GenerationDataError("训练未能产生有效权重")
            model.load_state_dict(best_state)
            test_loss = self._evaluate_loss(model, test_ids, loss_function, len(vocab))
            test_accuracy = self._evaluate_accuracy(model, test_ids, len(vocab))

            self.jobs.update(job_id, status="saving", stage="保存最佳权重、词表与测试结果")
            staging.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), staging / WEIGHT_FILE)
            (staging / VOCAB_FILE).write_text(
                json.dumps({"tokens": vocab}, ensure_ascii=False), encoding="utf-8"
            )
            metadata = {
                "id": job_id,
                "name": name,
                "algorithm": "char-transformer",
                "algorithm_label": f"字符级 Decoder-only Transformer（{WIDTH} 维 / {LAYERS} 层 / {HEADS} 头）",
                "created_at": utc_now(),
                "data_fingerprint": fingerprint,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "history": self.jobs.get(job_id)["history"],
                "test": {
                    "loss": round(test_loss, 4),
                    "perplexity": round(math.exp(min(test_loss, 20)), 2),
                    "next_char_accuracy": round(test_accuracy, 4),
                },
                "vocab_size": len(vocab),
                "context": CONTEXT,
                "width": WIDTH,
                "layers": LAYERS,
                "heads": HEADS,
                "document_count": len(documents),
                "char_count": sum(len(document) for document in documents),
                "train_seconds": round(time.perf_counter() - started, 2),
                "model_bytes": file_tree_bytes(staging),
                "param_count": sum(parameter.numel() for parameter in model.parameters()),
                "train_count": len(train_docs),
                "validation_count": len(validation_docs),
                "test_count": len(test_docs),
            }
            self.store.commit(staging, job_id, metadata)
            self.jobs.update(
                job_id, status="completed", stage="训练完成，模型已保存", model=metadata
            )
        finally:
            self.store.discard(staging)

    @staticmethod
    def _evaluate_loss(
        model: nn.Module, ids: list[int], loss_function: nn.Module, vocab_size: int
    ) -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(ids) - 1, CONTEXT - 1):
                chunk = ids[start:start + CONTEXT]
                if len(chunk) < 2:
                    break
                source = torch.tensor([chunk[:-1]], dtype=torch.long)
                target = torch.tensor([chunk[1:]], dtype=torch.long)
                logits = model(source)
                loss = loss_function(logits.reshape(-1, vocab_size), target.reshape(-1))
                total += loss.item()
                count += 1
        return total / max(count, 1)

    @staticmethod
    def _evaluate_accuracy(model: nn.Module, ids: list[int], vocab_size: int) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, len(ids) - 1, CONTEXT - 1):
                chunk = ids[start:start + CONTEXT]
                if len(chunk) < 2:
                    break
                source = torch.tensor([chunk[:-1]], dtype=torch.long)
                target = torch.tensor([chunk[1:]], dtype=torch.long)
                predictions = model(source).argmax(dim=-1)[0]
                correct += int((predictions == target[0]).sum())
                total += len(target[0])
        return correct / max(total, 1)

    # ---------- 推理 ----------

    def _load(self, model_id: str) -> None:
        with self._lock:
            folder = self.store.folder(model_id)
            if self._cached_id == model_id:
                return
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            vocab = json.loads((folder / VOCAB_FILE).read_text(encoding="utf-8"))["tokens"]
            model = CausalDecoder(len(vocab), metadata["width"], metadata["layers"], metadata["heads"])
            model.load_state_dict(
                torch.load(folder / WEIGHT_FILE, map_location="cpu", weights_only=True)
            )
            model.eval()
            self._cached_id = model_id
            self._cached_model = model
            self._cached_vocab = vocab
            self._cached_index = {token: index for index, token in enumerate(vocab)}
            self._cached_metadata = metadata

    def generate(
        self, model_id: str, prompt: str, max_new_tokens: int, temperature: float, top_k: int
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        if not 1 <= len(prompt) <= 200:
            raise GenerationDataError("开头需为 1–200 个字符")
        if not 10 <= max_new_tokens <= 120:
            raise GenerationDataError("生成长度需在 10–120 之间")
        if not 0.2 <= temperature <= 1.5:
            raise GenerationDataError("温度需在 0.2–1.5 之间")
        if not 1 <= top_k <= 50:
            raise GenerationDataError("top-k 需在 1–50 之间")

        with self._lock:
            self._load(model_id)
            assert self._cached_model is not None and self._cached_vocab is not None
            assert self._cached_index is not None
            generator = torch.Generator().manual_seed(SEED)
            sequence = [BOS] + [self._cached_index.get(character, UNK) for character in prompt]
            produced: list[str] = []
            stopped = False
            with torch.no_grad():
                for step in range(max_new_tokens):
                    window = torch.tensor([sequence[-CONTEXT:]], dtype=torch.long)
                    logits = self._cached_model(window)[0, -1].clone()
                    logits[PAD] = float("-inf")
                    logits[UNK] = float("-inf")
                    logits[BOS] = float("-inf")
                    if step < 5:
                        logits[EOS] = float("-inf")
                    values, indices = torch.topk(logits, min(top_k, logits.numel()))
                    probabilities = torch.softmax(values / temperature, dim=0)
                    choice = int(torch.multinomial(probabilities, 1, generator=generator).item())
                    token_id = int(indices[choice].item())
                    if token_id == EOS:
                        stopped = True
                        break
                    sequence.append(token_id)
                    produced.append(self._cached_vocab[token_id])
            return {
                "data": {
                    "text": "".join(produced),
                    "full_text": prompt + "".join(produced),
                    "generated_chars": len(produced),
                    "stopped_by_eos": stopped,
                },
                "model_id": model_id,
                "notice": "语料规模有限，续写内容仅用于演示，可能存在事实错误或不连贯。",
            }

    # ---------- 删除 ----------

    def delete(self, model_id: str) -> None:
        with self._lock:
            self.store.folder(model_id)
            if self._cached_id == model_id:
                self._cached_id = None
                self._cached_model = None
                self._cached_vocab = None
                self._cached_index = None
                self._cached_metadata = None
            self.store.delete(model_id)


generations = GenerationService(Path(os.getenv(
    "GENERATION_MODEL_DIR",
    str(Path(__file__).resolve().parents[2] / "artifacts" / "generation"),
)))
