#!/bin/bash

set -e

CONDA_CHANNELS="${CONDA_CHANNELS:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/,https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

CONTAINER_NAME="credit-model-dev"
ENV_NAME="model"
PYTHON_VERSION="3.11"
REQUIREMENTS_FILE="./requirements.txt"

echo "正在启动开发容器..."
docker-compose up -d

# 等待容器就绪
sleep 2

echo "正在容器中创建 Conda 环境: $ENV_NAME (Python $PYTHON_VERSION)..."

docker exec -i "$CONTAINER_NAME" bash -c '
  IFS=, read -ra CHANNELS <<< "$1"
  CMD=(conda create -n '"$ENV_NAME"' python='"$PYTHON_VERSION"' -y --override-channels)
  for ch in "${CHANNELS[@]}"; do
    CMD+=(-c "$ch")
  done
  "${CMD[@]}"
' _ "$CONDA_CHANNELS"

echo "正在安装依赖（$REQUIREMENTS_FILE）..."

if [ -f "$REQUIREMENTS_FILE" ]; then
  docker exec -i "$CONTAINER_NAME" bash -c '
    set -e
    eval "$(conda shell.bash hook)"
    conda activate '"$ENV_NAME"'
    env PYTHONUNBUFFERED=1 pip install -r /app/requirements.txt -i "$1"
  ' _ "$PIP_INDEX_URL"
else
  echo "警告: $REQUIREMENTS_FILE 不存在，跳过依赖安装。"
fi

echo "正在验证环境..."
docker exec -i "$CONTAINER_NAME" bash -c "
  conda run -n $ENV_NAME python -c \"import pandas; print('pandas 已成功导入，环境初始化完成！')\"
"

echo "开发环境已就绪！"
echo "进入容器调试命令："
echo "   docker exec -it $CONTAINER_NAME bash"
echo "在容器中激活环境（交互式 shell 中）："
echo "   conda activate $ENV_NAME"