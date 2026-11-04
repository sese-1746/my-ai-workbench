"""模块D 问答检索：TF-IDF 双通道（词 + 字符）检索。

没有神经网络权重，但同样遵循统一的任务/制品约定：
每一「轮」训练 = 评估一组检索配置（词 n-gram / 字符 n-gram / 融合权重），
按验证集 MRR 选出最优配置，再用留出测试集给出 Recall@1、Recall@3 与命中矩阵。
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
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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
MIN_PAIRS, MAX_PAIRS = 10, 2000
MAX_QUESTION_CHARS = 300
MAX_ANSWER_CHARS = 2000
INDEX_FILE = "index.joblib"
TOP_K_METRIC = 3

# 每轮搜索一组配置；epochs 决定搜索前 N 组
CONFIGS: list[dict[str, Any]] = [
    {"word_ngram": [1, 1], "char_ngram": [2, 4], "word_weight": 0.5},
    {"word_ngram": [1, 2], "char_ngram": [2, 4], "word_weight": 0.6},
    {"word_ngram": [1, 2], "char_ngram": [2, 5], "word_weight": 0.5},
    {"word_ngram": [1, 2], "char_ngram": [3, 5], "word_weight": 0.7},
    {"word_ngram": [1, 1], "char_ngram": [2, 5], "word_weight": 0.4},
    {"word_ngram": [1, 3], "char_ngram": [2, 4], "word_weight": 0.6},
    {"word_ngram": [1, 2], "char_ngram": [2, 3], "word_weight": 0.5},
    {"word_ngram": [1, 1], "char_ngram": [3, 4], "word_weight": 0.6},
]

QUESTION_KEYS = ("question", "q", "query", "问题", "提问")
ANSWER_KEYS = ("answer", "a", "response", "答案", "回答", "回复")
PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)
LATIN_TOKEN = re.compile(r"[a-zA-Z0-9']+")


class QaDataError(DataError):
    """问答数据或查询参数不合法。"""


def chinese_tokenizer(text: str) -> list[str]:
    """jieba 分词 + 英文/数字片段兜底，过滤纯标点。"""
    lowered = text.lower()
    tokens: list[str] = []
    for piece in jieba.lcut(lowered):
        piece = piece.strip()
        if not piece or PUNCT_ONLY.match(piece):
            continue
        if LATIN_TOKEN.fullmatch(piece) or re.search(r"[\u4e00-\u9fff]", piece):
            tokens.append(piece)
    return tokens


def parse_dataset(raw: bytes, file_type: str) -> tuple[list[str], list[str]]:
    if not raw:
        raise QaDataError("问答文件为空，请重新选择")
    if len(raw) > MAX_BYTES:
        raise QaDataError("问答文件不能超过 5 MB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise QaDataError("文件需使用 UTF-8 编码") from exc

    pairs: list[tuple[str, str]] = []
    if file_type == "jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QaDataError(f"第 {line_no} 行不是合法 JSON") from exc
            if not isinstance(item, dict):
                raise QaDataError(f"第 {line_no} 行必须是 JSON 对象")
            question = _pick(item, QUESTION_KEYS)
            answer = _pick(item, ANSWER_KEYS)
            pairs.append((question.strip(), answer.strip()))
    else:
        try:
            frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            raise QaDataError("请上传 UTF-8 编码、带表头的 CSV") from exc
        question_column = _find_column(frame.columns, QUESTION_KEYS)
        answer_column = _find_column(frame.columns, ANSWER_KEYS)
        if question_column is None or answer_column is None:
            raise QaDataError("CSV 需要 question/answer 两列（也接受 问题/答案）")
        for _, row in frame.iterrows():
            question = "" if pd.isna(row[question_column]) else str(row[question_column]).strip()
            answer = "" if pd.isna(row[answer_column]) else str(row[answer_column]).strip()
            pairs.append((question, answer))

    for question, answer in pairs:
        if not question or not answer:
            raise QaDataError("每条记录都必须同时有问题和答案")
        if len(question) > MAX_QUESTION_CHARS:
            raise QaDataError(f"问题不能超过 {MAX_QUESTION_CHARS} 字")
        if len(answer) > MAX_ANSWER_CHARS:
            raise QaDataError(f"答案不能超过 {MAX_ANSWER_CHARS} 字")
    if not MIN_PAIRS <= len(pairs) <= MAX_PAIRS:
        raise QaDataError(f"问答对需在 {MIN_PAIRS}–{MAX_PAIRS} 组之间，当前 {len(pairs)} 组")
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _pick(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _find_column(columns, keys: tuple[str, ...]) -> str | None:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for key in keys:
        if key in lowered:
            return lowered[key]
    return None


class RetrievalIndex:
    """双通道 TF-IDF 索引：词级抓关键词，字符级抓近义改写。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.word_vectorizer = TfidfVectorizer(
            tokenizer=chinese_tokenizer,
            lowercase=True,
            token_pattern=None,
            ngram_range=tuple(config["word_ngram"]),
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=tuple(config["char_ngram"]),
            sublinear_tf=True,
        )
        self.questions: list[str] = []
        self.answers: list[str] = []
        self.word_matrix = None
        self.char_matrix = None

    def fit(self, questions: list[str], answers: list[str]) -> "RetrievalIndex":
        """拟合词表并索引这批答案。"""
        # 拟合语料同时包含问题与答案，保证答案侧的专有名词也有向量
        self.word_vectorizer.fit(questions + answers)
        self.char_vectorizer.fit(questions + answers)
        return self.index(questions, answers)

    def index(self, questions: list[str], answers: list[str]) -> "RetrievalIndex":
        """只重建答案矩阵，复用已拟合的词表（验证/测试集用）。"""
        self.questions = questions
        self.answers = answers
        self.word_matrix = self.word_vectorizer.transform(answers)
        self.char_matrix = self.char_vectorizer.transform(answers)
        return self

    def clone(self) -> "RetrievalIndex":
        """复制一个共享词表、但可独立索引的实例。"""
        twin = RetrievalIndex(self.config)
        twin.word_vectorizer = self.word_vectorizer
        twin.char_vectorizer = self.char_vectorizer
        return twin

    def scores(self, query: str) -> np.ndarray:
        word_vector = self.word_vectorizer.transform([query])
        char_vector = self.char_vectorizer.transform([query])
        weight = float(self.config["word_weight"])
        word_scores = (self.word_matrix @ word_vector.T).toarray().ravel()
        char_scores = (self.char_matrix @ char_vector.T).toarray().ravel()
        return weight * word_scores + (1 - weight) * char_scores


def rank_metrics(scores: np.ndarray, gold_index: int) -> tuple[int, float]:
    """返回 (命中排名, 倒数排名)；排名从 1 开始。"""
    order = np.argsort(-scores)
    position = int(np.where(order == gold_index)[0][0]) + 1
    return position, 1.0 / position


def evaluate(index: RetrievalIndex, questions: list[str], gold_indices: list[int]) -> dict[str, Any]:
    recall1 = recall3 = 0
    reciprocal = 0.0
    true_positive = false_negative = false_positive = 0
    total_pairs = len(questions) * len(index.answers)
    for question, gold in zip(questions, gold_indices):
        scores = index.scores(question)
        position, rr = rank_metrics(scores, gold)
        recall1 += int(position == 1)
        recall3 += int(position <= TOP_K_METRIC)
        reciprocal += rr
        # 以 top-3 为"预测相关"，统计 query×answer 的 2×2 命中矩阵
        top = set(np.argsort(-scores)[:TOP_K_METRIC].tolist())
        true_positive += int(gold in top)
        false_negative += int(gold not in top)
        false_positive += len(top - {gold})
    true_negative = max(total_pairs - true_positive - false_negative - false_positive, 0)
    count = max(len(questions), 1)
    return {
        "recall_at_1": round(recall1 / count, 4),
        "recall_at_3": round(recall3 / count, 4),
        "mrr": round(reciprocal / count, 4),
        "confusion_matrix": [[true_positive, false_negative], [false_positive, true_negative]],
        "confusion_labels": ["相关", "不相关"],
        "metric_note": f"以 top-{TOP_K_METRIC} 为预测相关，按 query×answer 全量配对统计",
    }


class QaService:
    """检索器的训练（配置搜索）、持久化与查询。"""

    def __init__(self, root: Path) -> None:
        self.store = ModelStore(root, (INDEX_FILE,), "问答模型")
        self.jobs = JobRegistry("qa-train", "已有问答模型正在训练，请等待当前任务结束后再提交")
        self._lock = self.jobs.lock
        self._cached_id: str | None = None
        self._cached_index: RetrievalIndex | None = None
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
            raise QaDataError("模型名称需为 1–80 个字符")
        if not 1 <= epochs <= len(CONFIGS):
            raise QaDataError(f"配置搜索轮数需在 1–{len(CONFIGS)} 之间")
        questions, answers = parse_dataset(raw, file_type)
        return self.jobs.submit(
            epochs=epochs,
            worker=lambda job_id: self._train(
                job_id, questions, answers, name, epochs, sha256_hex(raw)
            ),
            pair_count=len(questions),
        )

    def _train(
        self,
        job_id: str,
        questions: list[str],
        answers: list[str],
        name: str,
        epochs: int,
        fingerprint: str,
    ) -> None:
        staging = self.store.staging(job_id)
        started = time.perf_counter()
        try:
            self.jobs.update(job_id, status="training", stage="划分数据并构建检索索引")
            indices = np.arange(len(questions))
            train_val_idx, test_idx = train_test_split(
                indices, test_size=0.15, random_state=SEED, shuffle=True
            )
            train_idx, validation_idx = train_test_split(
                train_val_idx, test_size=0.1765, random_state=SEED, shuffle=True
            )

            train_questions = [questions[i] for i in train_idx]
            train_answers = [answers[i] for i in train_idx]
            validation_questions = [questions[i] for i in validation_idx]
            validation_answers = [answers[i] for i in validation_idx]
            test_questions = [questions[i] for i in test_idx]
            test_answers = [answers[i] for i in test_idx]
            validation_gold = list(range(len(validation_questions)))
            test_gold = list(range(len(test_questions)))

            best_mrr, best_epoch, best_config = -1.0, 0, None
            for epoch, config in enumerate(CONFIGS[:epochs], start=1):
                # 词表只在训练集上拟合，验证时仅替换答案库，避免信息泄漏
                base = RetrievalIndex(config).fit(train_questions, train_answers)
                validation_index = base.clone().index(validation_questions, validation_answers)
                metrics = evaluate(validation_index, validation_questions, validation_gold)
                entry = {
                    "epoch": epoch,
                    "config": config,
                    "validation_recall_at_1": metrics["recall_at_1"],
                    "validation_recall_at_3": metrics["recall_at_3"],
                    "validation_mrr": metrics["mrr"],
                }
                if metrics["mrr"] > best_mrr:
                    best_mrr, best_epoch, best_config = metrics["mrr"], epoch, config
                self.jobs.record_epoch(
                    job_id, entry, epoch, epochs,
                    f"第 {epoch}/{epochs} 轮配置，验证 MRR {metrics['mrr']:.4f}",
                )

            if best_config is None:
                raise QaDataError("未能找到可用的检索配置")
            # 正式模型：训练集 + 验证集一起入索引，测试集只在评估时使用
            final_index = RetrievalIndex(best_config).fit(
                [questions[i] for i in train_val_idx],
                [answers[i] for i in train_val_idx],
            )
            test_index = final_index.clone().index(test_questions, test_answers)
            test = evaluate(test_index, test_questions, test_gold)

            self.jobs.update(job_id, status="saving", stage="保存最优索引与测试结果")
            staging.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "config": best_config,
                    "word_vectorizer": final_index.word_vectorizer,
                    "char_vectorizer": final_index.char_vectorizer,
                    "word_matrix": final_index.word_matrix,
                    "char_matrix": final_index.char_matrix,
                    "questions": final_index.questions,
                    "answers": final_index.answers,
                },
                staging / INDEX_FILE,
            )
            metadata = {
                "id": job_id,
                "name": name,
                "algorithm": "tfidf-dual",
                "algorithm_label": "TF-IDF 双通道检索（词 n-gram + 字符 n-gram 加权融合）",
                "created_at": utc_now(),
                "data_fingerprint": fingerprint,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_config": best_config,
                "history": self.jobs.get(job_id)["history"],
                "test": test,
                "pair_count": len(questions),
                "train_count": len(train_idx),
                "validation_count": len(validation_idx),
                "test_count": len(test_idx),
                "answer_count": len(final_index.answers),
                "train_seconds": round(time.perf_counter() - started, 2),
                "model_bytes": file_tree_bytes(staging),
                "param_count": 0,
                "param_note": "检索模型无神经网络参数，参数量记为 0",
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
            payload = joblib.load(folder / INDEX_FILE)
            index = RetrievalIndex(payload["config"])
            index.word_vectorizer = payload["word_vectorizer"]
            index.char_vectorizer = payload["char_vectorizer"]
            index.word_matrix = payload["word_matrix"]
            index.char_matrix = payload["char_matrix"]
            index.questions = payload["questions"]
            index.answers = payload["answers"]
            self._cached_id = model_id
            self._cached_index = index
            self._cached_metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))

    def query(self, model_id: str, question: str, top_k: int) -> dict[str, Any]:
        question = (question or "").strip()
        if not 1 <= len(question) <= MAX_QUESTION_CHARS:
            raise QaDataError(f"问题需为 1–{MAX_QUESTION_CHARS} 个字符")
        if not 1 <= top_k <= 10:
            raise QaDataError("top-k 需在 1–10 之间")
        with self._lock:
            self._load(model_id)
            assert self._cached_index is not None and self._cached_metadata is not None
            scores = self._cached_index.scores(question)
            order = np.argsort(-scores)[:top_k]
            results = [
                {
                    "rank": rank + 1,
                    "score": round(float(scores[index]), 4),
                    "question": self._cached_index.questions[int(index)],
                    "answer": self._cached_index.answers[int(index)],
                }
                for rank, index in enumerate(order)
            ]
            return {
                "data": {"results": results, "answer_count": len(self._cached_index.answers)},
                "model_id": model_id,
                "notice": "检索结果按字面相似度排序，不含语义推理，也未做事实核验，请人工确认。",
            }

    # ---------- 删除 ----------

    def delete(self, model_id: str) -> None:
        with self._lock:
            self.store.folder(model_id)
            if self._cached_id == model_id:
                self._cached_id = None
                self._cached_index = None
                self._cached_metadata = None
            self.store.delete(model_id)


qas = QaService(Path(os.getenv(
    "QA_MODEL_DIR",
    str(Path(__file__).resolve().parents[2] / "artifacts" / "qa"),
)))
