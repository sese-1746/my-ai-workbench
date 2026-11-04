"""FastAPI 入口：健康检查、模块状态与四个业务的五件套端点。

统一约定：
* 训练返回 ``202 + job``，重活交给 Service 内部的单线程池；
* 数据校验失败 → 422（中文 detail 可直接展示）；上传超限 → 413；扩展名不符 → 415；模型不存在 → 404；
* 推理响应一律带 ``notice`` 免责声明。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .common import DataError
from .generation_service import MAX_BYTES as MAX_GENERATION_BYTES
from .generation_service import GenerationDataError, generations
from .house_service import MAX_BYTES as MAX_HOUSE_BYTES
from .house_service import HouseDataError, houses
from .qa_service import MAX_BYTES as MAX_QA_BYTES
from .qa_service import QaDataError, qas
from .sentiment_service import MAX_BYTES as MAX_SENTIMENT_BYTES
from .sentiment_service import SentimentDataError, sentiments

VERSION = "1.0.0"

app = FastAPI(title="我的 AI 工作台 API", version=VERSION)


# --------------------------------------------------------------------------
# 请求模型
# --------------------------------------------------------------------------
class HousePredictRequest(BaseModel):
    model_id: str
    features: dict[str, Any] = Field(default_factory=dict)


class SentimentRequest(BaseModel):
    model_id: str
    text: str = Field(min_length=1, max_length=2000)


class GenerationRequest(BaseModel):
    model_id: str
    prompt: str = Field(min_length=1, max_length=200)
    max_new_tokens: int = Field(default=60, ge=10, le=120)
    temperature: float = Field(default=0.8, ge=0.2, le=1.5)
    top_k: int = Field(default=20, ge=1, le=50)


class QaQueryRequest(BaseModel):
    model_id: str
    question: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=3, ge=1, le=10)


# --------------------------------------------------------------------------
# 异常处理：把 Pydantic 的结构化错误压成一句可展示的中文
# --------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {"loc": (), "msg": "参数不合法"}
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = f"参数 {location} 不合法：{first.get('msg', '请检查输入')}" if location else first.get("msg", "参数不合法")
    return JSONResponse(status_code=422, content={"detail": message})


@app.exception_handler(DataError)
async def data_error_handler(request: Request, exc: DataError) -> JSONResponse:
    """兜底：所有业务校验异常（含任务占用）统一转 422，message 直接展示。"""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# --------------------------------------------------------------------------
# 通用上传处理
# --------------------------------------------------------------------------
async def _read_upload(file: UploadFile | None, suffixes: tuple[str, ...], max_bytes: int, too_large: str) -> bytes:
    if file is None:
        raise HTTPException(422, "请上传数据文件")
    filename = (file.filename or "").lower()
    if not filename.endswith(suffixes):
        raise HTTPException(415, f"仅支持 {' / '.join(suffixes)} 文件")
    raw = await file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(413, too_large)
    if not raw:
        raise HTTPException(422, "上传文件为空")
    return raw


async def _resolve_source(
    *,
    text: str,
    file: UploadFile | None,
    suffixes: tuple[str, ...],
    file_types: dict[str, str],
    default_type: str,
    max_bytes: int,
    too_large: str,
    empty_message: str,
) -> tuple[bytes, str]:
    """文件与粘贴文本二选一，返回 (原始字节, 数据格式)。"""
    if file is not None and text.strip():
        raise HTTPException(422, "请只选择粘贴文本或上传文件其中一种")
    if file is not None:
        raw = await _read_upload(file, suffixes, max_bytes, too_large)
        filename = (file.filename or "").lower()
        kind = next(value for suffix, value in file_types.items() if filename.endswith(suffix))
        return raw, kind
    if not text.strip():
        raise HTTPException(422, empty_message)
    raw = text.encode("utf-8")
    if len(raw) > max_bytes:
        raise HTTPException(413, too_large)
    return raw, default_type


# --------------------------------------------------------------------------
# 健康检查 / 模块总览
# --------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    counts = {
        "house": len(houses.list_models()),
        "sentiment": len(sentiments.list_models()),
        "generation": len(generations.list_models()),
        "qa": len(qas.list_models()),
    }
    active = {
        "house": houses.jobs.active_summary(),
        "sentiment": sentiments.jobs.active_summary(),
        "generation": generations.jobs.active_summary(),
        "qa": qas.jobs.active_summary(),
    }
    return {
        "status": "ok",
        "service": "api",
        "version": VERSION,
        "models": counts,
        "training": {key: value for key, value in active.items() if value},
    }


@app.get("/api/v1/modules")
def modules() -> dict[str, Any]:
    catalog = [
        ("house", "房价预测", "上传房价 CSV 训练 MLP，填写特征做估价", houses),
        ("sentiment", "情感分析", "中文评论分类，BiGRU / TextCNN 可切换", sentiments),
        ("generation", "文本生成", "字符级 Transformer，按开头续写", generations),
        ("qa", "问答检索", "TF-IDF 双通道检索，返回最相关答案", qas),
    ]
    return {
        "data": [
            {
                "id": key,
                "name": name,
                "page": name,
                "status": "ready",
                "description": description,
                "model_count": len(service.list_models()),
            }
            for key, name, description, service in catalog
        ]
    }


# --------------------------------------------------------------------------
# 模块A 房价预测
# --------------------------------------------------------------------------
@app.get("/api/v1/house/models")
def house_models() -> dict[str, Any]:
    return {"data": houses.list_models()}


@app.post("/api/v1/house/train", status_code=202)
async def house_train(
    file: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=80),
    epochs: int = Form(default=60, ge=1, le=200),
) -> dict[str, Any]:
    try:
        raw = await _read_upload(file, (".csv",), MAX_HOUSE_BYTES, "CSV 不能超过 10 MB")
        return await run_in_threadpool(houses.submit, raw, name, epochs)
    except HouseDataError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()


@app.get("/api/v1/house/jobs/{job_id}")
def house_job(job_id: str) -> dict[str, Any]:
    job = houses.job(job_id)
    if job is None:
        raise HTTPException(404, "训练任务不存在或服务已重启，请刷新模型列表")
    return job


@app.delete("/api/v1/house/models/{model_id}")
def house_delete(model_id: str) -> dict[str, Any]:
    try:
        houses.delete(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    return {"deleted": model_id}


@app.post("/api/v1/house/predict")
async def house_predict(request: HousePredictRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(houses.predict, request.model_id, request.features)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    except HouseDataError as exc:
        raise HTTPException(422, str(exc)) from exc


# --------------------------------------------------------------------------
# 模块B 情感分析
# --------------------------------------------------------------------------
@app.get("/api/v1/sentiment/models")
def sentiment_models() -> dict[str, Any]:
    return {"data": sentiments.list_models()}


@app.post("/api/v1/sentiment/train", status_code=202)
async def sentiment_train(
    text: str = Form(default=""),
    file: UploadFile | None = File(default=None),
    name: str = Form(..., min_length=1, max_length=80),
    algorithm: str = Form(default="bigru"),
    tokenizer: str = Form(default="jieba"),
    epochs: int = Form(default=8, ge=1, le=20),
) -> dict[str, Any]:
    try:
        raw, file_type = await _resolve_source(
            text=text,
            file=file,
            suffixes=(".csv", ".jsonl"),
            file_types={".csv": "csv", ".jsonl": "jsonl"},
            default_type="csv",
            max_bytes=MAX_SENTIMENT_BYTES,
            too_large="数据文件不能超过 5 MB",
            empty_message="请粘贴 CSV 数据或上传文件",
        )
        return await run_in_threadpool(
            sentiments.submit, raw, file_type, name, algorithm, tokenizer, epochs
        )
    except SentimentDataError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if file is not None:
            await file.close()


@app.get("/api/v1/sentiment/jobs/{job_id}")
def sentiment_job(job_id: str) -> dict[str, Any]:
    job = sentiments.job(job_id)
    if job is None:
        raise HTTPException(404, "训练任务不存在或服务已重启，请刷新模型列表")
    return job


@app.delete("/api/v1/sentiment/models/{model_id}")
def sentiment_delete(model_id: str) -> dict[str, Any]:
    try:
        sentiments.delete(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    return {"deleted": model_id}


@app.post("/api/v1/sentiment/classify")
async def sentiment_classify(request: SentimentRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(sentiments.classify, request.model_id, request.text)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    except SentimentDataError as exc:
        raise HTTPException(422, str(exc)) from exc


# --------------------------------------------------------------------------
# 模块C 文本生成
# --------------------------------------------------------------------------
@app.get("/api/v1/generation/models")
def generation_models() -> dict[str, Any]:
    return {"data": generations.list_models()}


@app.post("/api/v1/generation/train", status_code=202)
async def generation_train(
    text: str = Form(default=""),
    file: UploadFile | None = File(default=None),
    name: str = Form(..., min_length=1, max_length=80),
    epochs: int = Form(default=5, ge=1, le=20),
) -> dict[str, Any]:
    try:
        raw, file_type = await _resolve_source(
            text=text,
            file=file,
            suffixes=(".txt", ".csv"),
            file_types={".txt": "txt", ".csv": "csv"},
            default_type="txt",
            max_bytes=MAX_GENERATION_BYTES,
            too_large="语料文件不能超过 2 MB",
            empty_message="请粘贴语料或上传文件",
        )
        return await run_in_threadpool(generations.submit, raw, file_type, name, epochs)
    except GenerationDataError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if file is not None:
            await file.close()


@app.get("/api/v1/generation/jobs/{job_id}")
def generation_job(job_id: str) -> dict[str, Any]:
    job = generations.job(job_id)
    if job is None:
        raise HTTPException(404, "训练任务不存在或服务已重启，请刷新模型列表")
    return job


@app.delete("/api/v1/generation/models/{model_id}")
def generation_delete(model_id: str) -> dict[str, Any]:
    try:
        generations.delete(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    return {"deleted": model_id}


@app.post("/api/v1/generation/generate")
async def generation_generate(request: GenerationRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(
            generations.generate,
            request.model_id,
            request.prompt,
            request.max_new_tokens,
            request.temperature,
            request.top_k,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    except GenerationDataError as exc:
        raise HTTPException(422, str(exc)) from exc


# --------------------------------------------------------------------------
# 模块D 问答检索
# --------------------------------------------------------------------------
@app.get("/api/v1/qa/models")
def qa_models() -> dict[str, Any]:
    return {"data": qas.list_models()}


@app.post("/api/v1/qa/train", status_code=202)
async def qa_train(
    text: str = Form(default=""),
    file: UploadFile | None = File(default=None),
    name: str = Form(..., min_length=1, max_length=80),
    epochs: int = Form(default=4, ge=1, le=8),
) -> dict[str, Any]:
    try:
        raw, file_type = await _resolve_source(
            text=text,
            file=file,
            suffixes=(".csv", ".jsonl"),
            file_types={".csv": "csv", ".jsonl": "jsonl"},
            default_type="csv",
            max_bytes=MAX_QA_BYTES,
            too_large="问答文件不能超过 5 MB",
            empty_message="请粘贴 CSV 数据或上传文件",
        )
        return await run_in_threadpool(qas.submit, raw, file_type, name, epochs)
    except QaDataError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if file is not None:
            await file.close()


@app.get("/api/v1/qa/jobs/{job_id}")
def qa_job(job_id: str) -> dict[str, Any]:
    job = qas.job(job_id)
    if job is None:
        raise HTTPException(404, "训练任务不存在或服务已重启，请刷新模型列表")
    return job


@app.delete("/api/v1/qa/models/{model_id}")
def qa_delete(model_id: str) -> dict[str, Any]:
    try:
        qas.delete(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    return {"deleted": model_id}


@app.post("/api/v1/qa/query")
async def qa_query(request: QaQueryRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(qas.query, request.model_id, request.question, request.top_k)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    except QaDataError as exc:
        raise HTTPException(422, str(exc)) from exc
