"""跨业务复用的训练任务队列与模型制品存储。

四个业务模块共享同一套骨架：

* 内存任务字典 + ``RLock`` + 单线程池：同一模块同时只允许一个活跃训练任务；
* 状态机 ``queued → training → saving → completed / failed``，逐轮把指标追加到 ``history``；
* 制品先写入 ``.{uuid}.tmp`` 暂存目录，写完 ``metadata.json`` 后整体 ``rename``，
  保证 ``list_models`` 永远看不到半成品；
* 模型目录名必须是 UUID，且目录内文件齐全、``metadata["id"]`` 与目录名自洽。
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

# 活跃状态：处于这些状态时拒绝提交新任务
ACTIVE_STATUSES = ("queued", "training", "saving")

# 所有随机源统一使用的种子
SEED = 42


class DataError(ValueError):
    """业务数据校验失败。

    main.py 统一把它转换成 HTTP 422，``str(exc)`` 是可直接展示的中文文案。
    """


class JobBusyError(DataError):
    """同一模块已有活跃训练任务。"""


def utc_now() -> str:
    """返回秒级 ISO-8601 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_hex(raw: bytes) -> str:
    """数据指纹：用于横向比较"同一份数据"训练出的模型。"""
    return hashlib.sha256(raw).hexdigest()


def file_tree_bytes(folder: Path, skip: tuple[str, ...] = ("metadata.json",)) -> int:
    """统计目录内文件总字节数（默认跳过 metadata.json 自身）。"""
    total = 0
    for path in folder.rglob("*"):
        if path.is_file() and path.name not in skip:
            total += path.stat().st_size
    return total


class JobRegistry:
    """单模块的训练任务注册表：内存态、单并发、逐轮进度。"""

    def __init__(self, thread_prefix: str, busy_message: str) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_prefix)
        self.busy_message = busy_message

    # ---------- 对外查询 ----------

    @property
    def lock(self) -> threading.RLock:
        """暴露可重入锁，供业务类的推理缓存共用（避免嵌套加锁顺序不一致）。"""
        return self._lock

    def has_active(self) -> bool:
        with self._lock:
            return any(job["status"] in ACTIVE_STATUSES for job in self._jobs.values())

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def active_summary(self) -> dict[str, Any] | None:
        """返回当前活跃任务的精简信息，供前端提示"正在训练"。"""
        with self._lock:
            for job in self._jobs.values():
                if job["status"] in ACTIVE_STATUSES:
                    return {"id": job["id"], "status": job["status"], "stage": job["stage"]}
        return None

    # ---------- 生命周期 ----------

    def submit(self, *, epochs: int, worker: Callable[[str], None], **summary: Any) -> dict[str, Any]:
        """登记任务并投递到单线程池；已有活跃任务时抛 ``JobBusyError``。"""
        with self._lock:
            if self.has_active():
                raise JobBusyError(self.busy_message)
            job_id = str(uuid4())
            self._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "stage": "数据校验完成，等待训练",
                "current_epoch": 0,
                "epochs": epochs,
                "history": [],
                "model": None,
                "error": None,
                **summary,
            }
            self._executor.submit(self._run, job_id, worker)
            return copy.deepcopy(self._jobs[job_id])

    def _run(self, job_id: str, worker: Callable[[str], None]) -> None:
        try:
            worker(job_id)
        except Exception as exc:  # noqa: BLE001 - 任务线程必须兜底，否则任务永远停在 training
            detail = str(exc).strip() or exc.__class__.__name__
            self.fail(job_id, detail)
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job["status"] in ACTIVE_STATUSES:
                # worker 正常返回却没走到 completed，视为失败，避免假死
                job.update(status="failed", stage="训练失败", error="训练未正常结束，请重试")

    def update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def fail(self, job_id: str, message: str) -> None:
        with self._lock:
            self._jobs[job_id].update(status="failed", stage="训练失败", error=f"训练失败：{message}")

    def record_epoch(self, job_id: str, entry: dict[str, Any], epoch: int, epochs: int, stage: str) -> None:
        """追加一轮指标并同步 current_epoch / stage。"""
        with self._lock:
            job = self._jobs[job_id]
            job["history"].append(entry)
            job["current_epoch"] = epoch
            job["stage"] = stage


class ModelStore:
    """模型制品目录：原子写入、合法性过滤、删除。"""

    def __init__(self, root: Path, required_files: tuple[str, ...], model_label: str) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.required_files = required_files
        self.model_label = model_label

    # ---------- 查询 ----------

    def list_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for metadata_path in self.root.glob("*/metadata.json"):
            folder = metadata_path.parent
            if folder.name.startswith("."):
                continue
            if not self._is_uuid(folder.name):
                continue
            if not all((folder / name).is_file() for name in self.required_files):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if metadata.get("id") != folder.name or "created_at" not in metadata:
                continue
            models.append(metadata)
        return sorted(models, key=lambda item: item["created_at"], reverse=True)

    def exists(self, model_id: str) -> bool:
        try:
            self.folder(model_id)
        except KeyError:
            return False
        return True

    def folder(self, model_id: str) -> Path:
        """返回合法模型目录，不存在或残缺时抛 ``KeyError``（main.py 转 404）。"""
        if not self._is_uuid(model_id):
            raise KeyError(f"{self.model_label}不存在")
        folder = self.root / model_id
        if not all((folder / name).is_file() for name in self.required_files):
            raise KeyError(f"{self.model_label}不存在")
        return folder

    def metadata(self, model_id: str) -> dict[str, Any]:
        path = self.folder(model_id) / "metadata.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise KeyError(f"{self.model_label}不存在") from exc

    # ---------- 写入 ----------

    def staging(self, job_id: str) -> Path:
        """暂存目录，与正式目录同盘，保证 rename 原子性。"""
        return self.root / f".{job_id}.tmp"

    def commit(self, staging: Path, model_id: str, metadata: dict[str, Any]) -> Path:
        """写入 metadata 后整体改名为正式目录。"""
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        target = self.root / model_id
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
        return target

    def discard(self, staging: Path) -> None:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    # ---------- 删除 ----------

    def delete(self, model_id: str) -> None:
        shutil.rmtree(self.folder(model_id))

    @staticmethod
    def _is_uuid(value: str) -> bool:
        try:
            return str(UUID(value)) == value
        except (ValueError, AttributeError, TypeError):
            return False
