# synth/build.py -- синтетические задачи (synth/tasks/*.py) -> формат конкурсных задач для песочницы
"""
Каждая задача описана словарём в списке TASKS модулей synth/tasks/*.py:
  name      короткое имя репозитория (instance_id = synth_<name>, repo = synth/<name>);
  topic     предметная область, kind -- "bug" или "feature";
  issue     условие задачи, как в issue на GitHub (то, что видит модель);
  files     {путь: текст} -- репозиторий до исправления, включая существующие тесты;
  fix       [(путь, старый текст, новый текст)] -- эталонная правка; старый текст встречается в файле ровно раз;
  tests     {путь: текст} -- скрытые тесты целиком (новые файлы или замена существующих);
  solution  {"input", "locate", "change", "verify", "output"} -- как модель должна решать: вход -> шаги -> выход.

Сборка пишет в --out то же, что лежит в data/comp_full: tasks.jsonl, snapshots/<instance_id>.tgz
(репозиторий с .git, базовый коммит), а также valid.json (задачи, прошедшие проверку), solutions.jsonl
и TASKS.md. Проверка на хосте: существующие тесты проходят до правки, скрытые падают без эталонной правки
и проходят с ней (вместе со всеми остальными тестами).

Песочница: python -m sim.mimo_server --comp data/synth --valid data/synth/valid.json
Запуск:    python -m synth.build --out data/synth [--only name ...] [--no-check]
"""
import argparse
import importlib
import json
import os
import pkgutil
import shutil
import subprocess
import sys
import tarfile
import tempfile

import synth.tasks

GIT_ENV = {"GIT_AUTHOR_NAME": "synth", "GIT_AUTHOR_EMAIL": "synth@example.com",
           "GIT_COMMITTER_NAME": "synth", "GIT_COMMITTER_EMAIL": "synth@example.com",
           "GIT_AUTHOR_DATE": "2026-10-01T12:00:00+00:00", "GIT_COMMITTER_DATE": "2026-10-01T12:00:00+00:00"}
CREATED_AT = "2026-10-01T12:00:00Z"
REQUIRED = ("name", "topic", "kind", "issue", "files", "fix", "tests", "solution")


def load_tasks(only=None):
    tasks = []
    for mod in sorted(pkgutil.iter_modules(synth.tasks.__path__), key=lambda m: m.name):
        tasks.extend(importlib.import_module(f"synth.tasks.{mod.name}").TASKS)
    return [t for t in tasks if not only or t["name"] in only]


def validate(task):
    """Структурные ошибки описания (без запуска тестов) -> список строк."""
    errors = [f"нет поля {k}" for k in REQUIRED if k not in task]
    if errors:
        return errors
    for path, old, _ in task["fix"]:
        text = task["files"].get(path)
        if text is None:
            errors.append(f"fix: файла {path} нет в files")
        elif text.count(old) != 1:
            errors.append(f"fix: в {path} старый текст встречается {text.count(old)} раз")
    for path in task["tests"]:
        if not os.path.basename(path).startswith("test_"):
            errors.append(f"tests: {path} не похож на тестовый файл")
    return errors


def write_files(root, files):
    for path, text in files.items():
        p = os.path.join(root, *path.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def apply_fix(root, fix):
    for path, old, new in fix:
        p = os.path.join(root, *path.split("/"))
        with open(p, encoding="utf-8") as f:
            text = f.read()
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text.replace(old, new, 1))


def git(root, *args):
    env = dict(os.environ, **GIT_ENV)
    return subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-C", root, *args],
                          check=True, capture_output=True, text=True, encoding="utf-8", env=env).stdout


def staged_diff(root):
    git(root, "add", "-A")
    return git(root, "diff", "--cached", "--no-color", "--no-ext-diff", "--no-renames")


def make_repo(root, task):
    """Репозиторий с базовым коммитом -> (sha, patch, test_patch); рабочее дерево остаётся на базе."""
    write_files(root, task["files"])
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", f"{task['name']}: initial import")
    sha = git(root, "rev-parse", "HEAD").strip()
    apply_fix(root, task["fix"])
    patch = staged_diff(root)
    git(root, "reset", "-q", "--hard")
    write_files(root, task["tests"])
    test_patch = staged_diff(root)
    git(root, "reset", "-q", "--hard")
    git(root, "clean", "-qfdx")
    return sha, patch, test_patch


def run_pytest(root, paths):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([root] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths], cwd=root, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0, r.stdout[-1500:] + r.stderr[-500:]


def check(task, tmp):
    """Три прогона на копиях репозитория -> None или текст ошибки."""
    existing = sorted(p for p in task["files"] if os.path.basename(p).startswith("test_"))
    hidden = sorted(task["tests"])
    runs = [("существующие тесты до правки", {}, False, existing, True),
            ("скрытые тесты без правки", task["tests"], False, hidden, False),
            ("все тесты с эталонной правкой", task["tests"], True, sorted(set(existing) | set(hidden)), True)]
    for k, (label, extra, fixed, paths, want) in enumerate(runs):
        if not paths:
            continue
        root = os.path.join(tmp, f"check{k}")
        write_files(root, task["files"])
        if fixed:
            apply_fix(root, task["fix"])
        write_files(root, extra)
        ok, out = run_pytest(root, paths)
        if ok != want:
            return f"{label}: ожидалось {'успех' if want else 'падение'}\n{out}"
    return None


def snapshot(root, path):
    with tarfile.open(path, "w:gz") as tar:
        tar.add(root, arcname=".")


def record(task, sha, patch, test_patch):
    return {"instance_id": f"synth_{task['name']}", "repo": f"synth/{task['name']}", "base_commit": sha,
            "patch": patch, "test_patch": test_patch, "problem_statement": task["issue"].strip() + "\n",
            "hints_text": "", "created_at": CREATED_AT, "verifier_timeout_sec": 120,
            "topic": task["topic"], "kind": task["kind"]}


def tasks_md(tasks, records):
    lines = ["# Синтетические задачи", "", "Сгенерировано `python -m synth.build`; правьте описания в `synth/tasks/`.", "",
             "| id | тема | тип | что изменить |", "|---|---|---|---|"]
    lines += [f"| `{r['instance_id']}` | {t['topic']} | {t['kind']} | {t['solution']['change']} |"
              for t, r in zip(tasks, records)]
    for t, r in zip(tasks, records):
        s = t["solution"]
        lines += ["", f"## {r['instance_id']}", "", f"**Вход:** {s['input']}", "", f"**Найти место:** {s['locate']}", "",
                  f"**Изменить:** {s['change']}", "", f"**Проверить:** `{s['verify']}`", "", f"**Выход:** {s['output']}"]
    return "\n".join(lines) + "\n"


def build(out, only=None, do_check=True, log=print):
    tasks = load_tasks(only)
    os.makedirs(os.path.join(out, "snapshots"), exist_ok=True)
    built, records, valid, failed = [], [], [], {}
    for task in tasks:
        errors = validate(task)
        if errors:
            failed[task.get("name", "?")] = "; ".join(errors)
            log(f"[описание] {task.get('name')}: {failed[task.get('name', '?')]}")
            continue
        tmp = tempfile.mkdtemp(prefix=f"synth_{task['name']}_")
        try:
            repo = os.path.join(tmp, "repo")
            sha, patch, test_patch = make_repo(repo, task)
            rec = record(task, sha, patch, test_patch)
            snapshot(repo, os.path.join(out, "snapshots", rec["instance_id"] + ".tgz"))
            problem = check(task, tmp) if do_check else None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        built.append(task)
        records.append(rec)
        if problem:
            failed[task["name"]] = problem
            log(f"[проверка] {rec['instance_id']}: {problem}")
        else:
            valid.append(rec["instance_id"])
            log(f"ok {rec['instance_id']} ({task['kind']}, {task['topic']})")
    with open(os.path.join(out, "tasks.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "valid.json"), "w", encoding="utf-8") as f:
        json.dump({"valid": valid, "checked": do_check}, f, indent=1)
    with open(os.path.join(out, "solutions.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for t, rec in zip(built, records):
            f.write(json.dumps({"instance_id": rec["instance_id"], **t["solution"]}, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "TASKS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(tasks_md(built, records))
    return valid, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--only", nargs="*", default=None, help="имена задач (поле name)")
    ap.add_argument("--no-check", action="store_true", help="не запускать pytest")
    args = ap.parse_args()
    valid, failed = build(args.out, args.only, not args.no_check)
    print(f"годных задач: {len(valid)}, с ошибками: {len(failed)} -> {args.out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
