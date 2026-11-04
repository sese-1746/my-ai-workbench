"""测试公共设施：隔离的模型目录、合成数据集、任务轮询。"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# 必须在导入 app 之前重定向模型目录，避免污染仓库内 artifacts/
TEST_ROOT = Path(tempfile.mkdtemp(prefix="ai-workbench-tests-"))
for env_name, folder in (
    ("HOUSE_MODEL_DIR", "house"),
    ("SENTIMENT_MODEL_DIR", "sentiment"),
    ("GENERATION_MODEL_DIR", "generation"),
    ("QA_MODEL_DIR", "qa"),
):
    os.environ[env_name] = str(TEST_ROOT / folder)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def wait_job(module: str, job_id: str, timeout: float = 180.0) -> dict:
    """轮询直到任务进入终态。"""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        response = client.get(f"/api/v1/{module}/jobs/{job_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in ("completed", "failed"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"任务 {job_id} 超时未结束，最后状态：{json.dumps(last, ensure_ascii=False)}")


def house_csv(rows: int = 120) -> bytes:
    rng = random.Random(7)
    lines = ["Id,Area,Rooms,Age,Neighborhood,SalePrice"]
    for index in range(rows):
        area = round(rng.uniform(60, 220), 1)
        rooms = rng.randint(1, 5)
        age = rng.randint(0, 40)
        neighborhood = ["A", "B", "C"][index % 3]
        price = round(area * 2600 + rooms * 12000 - age * 900 + {"A": 40000, "B": 0, "C": -20000}[neighborhood])
        lines.append(f"{index + 1},{area},{rooms},{age},{neighborhood},{max(price, 50000)}")
    return "\n".join(lines).encode("utf-8")


def sentiment_csv(per_class: int = 30) -> bytes:
    templates = {
        "正面": [
            "这款产品的质量非常好，用起来很顺手，客服也很耐心，值得推荐。",
            "物流很快，包装完好，价格实惠，家人都说买得值。",
            "第二次购买了，一如既往地满意，希望保持品质。",
        ],
        "负面": [
            "收到货就发现有破损，联系客服也没人回复，非常失望。",
            "质量太差，用了两天就坏了，完全是浪费时间。",
            "和描述完全不一样，颜色差别很大，不推荐购买。",
        ],
        "中性": [
            "东西一般，能用但没有惊喜，价格算是合理。",
            "包装普通，功能基本符合说明，暂时没发现问题。",
            "第一次买这个牌子，感觉中规中矩，再看看后续表现。",
        ],
    }
    lines = ["text,label"]
    for label, phrases in templates.items():
        for index in range(per_class):
            lines.append(f"{phrases[index % len(phrases)]}（第 {index + 1} 条）,{label}")
    return "\n".join(lines).encode("utf-8")


def generation_corpus(lines: int = 24) -> bytes:
    pool = [
        "人工智能正在改变我们的生活方式，从语音助手到自动驾驶，技术让许多想象成为现实。",
        "清晨的城市还没有完全醒来，街道上只有零星的脚步声和远处公交车的低鸣。",
        "学习写作最好的办法是持续阅读与练习，然后把想法写下来，再反复修改。",
        "数据可视化让复杂的信息变得直观，一张好的图表往往胜过千言万语。",
    ]
    documents = []
    for index in range(lines):
        base = pool[index % len(pool)]
        documents.append(f"{base}这是第{index}段语料，用于训练一个很小的语言模型。" * 2)
    return "\n".join(documents).encode("utf-8")


def qa_csv(pairs: int = 40) -> bytes:
    topics = [
        ("如何重置路由器", "长按机身背面的重置孔 10 秒，等待指示灯闪烁后松开，路由器会恢复出厂设置。"),
        ("怎么申请退款", "在订单详情页点击申请退款，填写原因并提交，审核通过后原路退回。"),
        ("发票在哪里下载", "登录账号后进入订单页面，选择开具发票，电子发票会发送到预留邮箱。"),
        ("会员如何续费", "在会员中心选择续费套餐，使用余额或在线支付完成续费即可。"),
        ("密码忘记了怎么办", "在登录页点击忘记密码，通过手机验证码验证身份后重新设置密码。"),
    ]
    lines = ["question,answer"]
    for index in range(pairs):
        question, answer = topics[index % len(topics)]
        lines.append(f"问题{index + 1}：{question}？,参考答案{index + 1}：{answer}")
    return "\n".join(lines).encode("utf-8")


def cleanup() -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
