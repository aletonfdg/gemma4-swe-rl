# Удалённая песочница MiMo (ноутбук с Docker -> Colab с GPU)

Colab не умеет Docker, поэтому инструменты агента и проверка тестов исполняются на машине с
Docker, а модель и обучение остаются в Colab. Код: `sim/mimo_sandbox.py` (контейнеры),
`sim/mimo_server.py` (HTTP), `sim/mimo_env.py` (клиент с интерфейсом `SWEEnv`).

## 1. На ноутбуке (Windows + Docker Desktop с WSL2)

```powershell
pip install datasets
python sim/pilot_mimo.py --data C:\Users\vishn\Desktop\data\mimo_v2_6_code --ids <id> ...   # один раз
python -m sim.mimo_server --data C:\Users\vishn\Desktop\data\mimo_v2_6_code --max-sessions 3
```
Сервер печатает токен и слушает только `127.0.0.1:8765`. Образы `xiaomimimo/mimo-v2.6-rl-oss:<id>`
Docker скачает сам при первом обращении к задаче (1-3 ГБ каждый, нужен интернет у ноутбука).

Туннель в соседнем окне (бесплатно, без регистрации):
```powershell
winget install Cloudflare.cloudflared
cloudflared tunnel --url http://127.0.0.1:8765
```
Скопируйте адрес `https://....trycloudflare.com`. Он меняется при каждом запуске. Токен не
публикуйте: по нему можно выполнять команды на ноутбуке (в контейнерах без сети).

## 2. В Colab

```python
import os
os.environ["SWE_SANDBOX_URL"] = "https://....trycloudflare.com"
os.environ["SWE_SANDBOX_TOKEN"] = "<токен с ноутбука>"
from sim.mimo_env import MimoRemoteEnv
env = MimoRemoteEnv()           # GET /tasks -> список instance_id
prompt = env.reset(0)           # запускает контейнер, возвращает промпт
```
Дальше `rl.rollout.run_episode(env, 0, llm_client, None, None)` работает как с `SWEEnv`.

## Что важно знать

- Награда = код возврата `bash mimo_test_command.sh` после применения `test_patch`; патч
  проверяется в СВЕЖЕМ контейнере, защищённые файлы (тесты, конфиги) перед этим откатываются.
- Эталонных патчей в датасете нет: что верное решение даёт `passed=True`, подтверждается только
  когда агент впервые решит задачу. Пока такого нет, считайте награду непроверенной.
- Код графа и эмбеддингов недоступен: `get_code_neighbors` и т. п. возвращают `GraphUnavailable`
  (в боевых задачах конкурса они есть, это расхождение между обучением и тестом).
- Параллельность ограничена `--max-sessions` (по умолчанию 3). Лишние `reset` ждут свободный слот.
- Долгие операции идут через `pending`/`poll`, чтобы туннель не обрывал запросы дольше ~100 с.
- Если ноутбук уснул или упал туннель, клиент помечает эпизод `env.crashed`; такие эпизоды не
  нужно брать в обучение как провалы агента.
