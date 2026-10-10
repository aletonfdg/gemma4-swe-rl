"""Colab: скачать из Kaggle только нужный минимум датасета gemma-4-developer-agent.

Использование в ячейках ноутбука:
    import os
    os.environ["KAGGLE_USERNAME"] = "ваш_логин"      # вписать в ячейке, НЕ в файле репозитория
    os.environ["KAGGLE_KEY"] = "ваш_api_ключ"
    !python colab_fetch_kaggle.py --per-repo 4
(либо загрузить kaggle.json через files.upload() -- тоже поддерживается)

Скачивает: tasks.jsonl, HARNESS_README.md, docker/, sandbox/setup.py и для выбранных задач
snapshots/, graphs/, embeddings/. wheels/ (124 файла) качаются только с --wheels.
Весь датасет (22 ГБ) НЕ скачивается.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict

COMP = "gemma-4-developer-agent"


def setup_kaggle_auth(username=None, key=None):
    """Приоритет: аргументы -> переменные окружения -> kaggle.json.
    Ключ НЕ хранить в коде, который коммитится: передавать из ячейки ноутбука."""
    if username and key:
        os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = username, key
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    dst = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(dst):
        for cand in ("kaggle.json", "/content/kaggle.json", "/content/drive/MyDrive/kaggle.json"):
            if os.path.exists(cand):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(cand, dst)
                break
        else:
            sys.exit("kaggle.json не найден: загрузите его в /content или задайте KAGGLE_USERNAME/KAGGLE_KEY")
    os.chmod(dst, 0o600)


def fetch(remote_path, data_dir):
    """Качает один файл соревнования в data_dir/<remote_path>. Пропускает уже скачанное."""
    local = os.path.join(data_dir, remote_path)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    out_dir = os.path.dirname(local)
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", COMP, "-f", remote_path,
                    "-p", out_dir, "-q"], check=True)
    # CLI иногда отдаёт <name>.zip -- распаковываем
    zipped = local + ".zip"
    if os.path.exists(zipped):
        with zipfile.ZipFile(zipped) as z:
            z.extractall(out_dir)
        os.remove(zipped)
    if not os.path.exists(local):
        sys.exit(f"Не удалось получить {remote_path} (проверьте, что правила конкурса приняты)")
    return local


def pick_tasks(tasks, per_repo):
    by_repo = defaultdict(list)
    for t in tasks:
        by_repo[t["repo"]].append(t)
    picked = []
    for repo, ts in sorted(by_repo.items()):
        ts.sort(key=lambda t: t["instance_id"])
        picked += ts[:per_repo]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--per-repo", type=int, default=4, help="задач на каждый из 4 репозиториев")
    ap.add_argument("--wheels", action="store_true", help="скачать все wheels (нужны для офлайн-установки)")
    ap.add_argument("--kaggle-username", default=None)
    ap.add_argument("--kaggle-key", default=None)
    args = ap.parse_args()

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    setup_kaggle_auth(args.kaggle_username, args.kaggle_key)
    d = args.data_dir

    for f in ("tasks.jsonl", "HARNESS_README.md", "docker/Dockerfile.sandbox",
              "docker/Dockerfile.public", "docker/imp.py", "docker/telnetlib.py",
              "sandbox/setup.py"):
        fetch(f, d)

    tasks = [json.loads(l) for l in open(os.path.join(d, "tasks.jsonl"), encoding="utf-8")]
    print(f"tasks.jsonl: {len(tasks)} задач")
    chosen = pick_tasks(tasks, args.per_repo)

    for t in chosen:
        iid = t["instance_id"]
        for sub, ext in (("snapshots", "tgz"), ("graphs", "json"), ("embeddings", "npz")):
            fetch(f"{sub}/{iid}.{ext}", d)
        print("OK", iid)

    if args.wheels:
        # список файлов wheels неизвестен заранее -- берём через CLI
        out = subprocess.run(["kaggle", "competitions", "files", "-c", COMP, "--page-size", "1000", "-v"],
                             capture_output=True, text=True, check=True).stdout
        names = [l.split(",")[0] for l in out.splitlines() if l.startswith("wheels/")]
        for n in names:
            fetch(n, d)

    with open(os.path.join(d, "selected_tasks.json"), "w") as f:
        json.dump([t["instance_id"] for t in chosen], f, indent=1)
    print(f"Готово: {len(chosen)} задач в {d}/ (список: {d}/selected_tasks.json)")


if __name__ == "__main__":
    main()
