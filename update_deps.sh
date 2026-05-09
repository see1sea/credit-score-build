#!/bin/bash
set -e

CONTAINER="credit-model-dev"
ENV="model"
REQ="/app/requirements.txt"

# 检查容器是否存在
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: Container '$CONTAINER' does not exist."
    echo "Please ensure your docker-compose.yml is in the current directory and run 'docker compose up -d' first."
    exit 1
fi

# 检查容器是否正在运行，若未运行则启动
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Container '$CONTAINER' is not running. Starting it now..."
    docker start "$CONTAINER"
    # 可选：等待几秒让服务稳定（如 Conda 环境加载）
    sleep 3
else
    echo "Container '$CONTAINER' is already running."
fi

echo "Checking if requirements file exists in container..."
if ! docker exec "$CONTAINER" test -f "$REQ"; then
    echo "ERROR: $REQ not found inside container."
    echo "Make sure your volume mount includes the requirements.txt file."
    exit 1
fi

echo "Installing dependencies with verbose output..."
docker exec "$CONTAINER" \
  env PYTHONUNBUFFERED=1 \
  conda run -n "$ENV" \
  pip install -v -r "$REQ"