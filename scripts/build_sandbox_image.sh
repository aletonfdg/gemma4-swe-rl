#!/bin/bash
# scripts/build_sandbox_image.sh
# Собирает swebench-sandbox:latest из docker/Dockerfile.sandbox (нужен для sim/container.py
# и для локального `swegemma eval --sandbox docker`, если CLI тоже доступен).
set -e

DATASET_ROOT="${1:-data}"

if [ ! -f "${DATASET_ROOT}/docker/Dockerfile.sandbox" ]; then
  echo "Не найден ${DATASET_ROOT}/docker/Dockerfile.sandbox -- проверьте путь к датасету."
  exit 1
fi

docker build \
  -f "${DATASET_ROOT}/docker/Dockerfile.sandbox" \
  -t swebench-sandbox:latest \
  "${DATASET_ROOT}/docker"

echo "Готово: образ swebench-sandbox:latest собран."
echo "Если отдельные тестовые окружения требуют Dockerfile.public (публичные 129 задач) --"
echo "собрать его аналогично под отдельным тегом и сверить, какой из двух реально"
echo "ожидает sim/container.py / реальный swegemma CLI (см. HARNESS_README.md раздел 4.1)."