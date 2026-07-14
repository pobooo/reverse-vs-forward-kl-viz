#!/usr/bin/env bash
# 启动 KL 可视化服务
set -e
cd "$(dirname "$0")/backend"

if [ ! -d ".venv" ]; then
  echo ">>> 创建 venv..."
  python3 -m venv .venv
fi
source .venv/bin/activate

echo ">>> 安装依赖..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ">>> 启动 FastAPI (http://127.0.0.1:8000)"
exec uvicorn app:app --host 127.0.0.1 --port 8000 --reload
