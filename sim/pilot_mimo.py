"""Пилот: можно ли на этой машине получить сигнал «тесты падают» из образов MiMo.

    python sim/pilot_mimo.py --data C:/Users/vishn/Desktop/data/mimo_v2_6_code \
        --ids format-code-task-001499 format-code-task-002366 [--pull] [--network]

Для каждой задачи: (опционально docker pull) -> контейнер из образа -> `git apply` test_patch
(он же создаёт mimo_test_command.sh) -> `bash mimo_test_command.sh` в cwd задачи.
Ожидание на НЕисправленном репозитории: код возврата != 0 (тесты падают).
Это проверяет только половину награды: что после верного патча тесты проходят, пилот
показать не может -- эталонных патчей в датасете нет.
Требуется только docker CLI и пакет `datasets`.
"""
import argparse
import json
import subprocess
import sys
import time

REGISTRY = "xiaomimimo/mimo-v2.6-rl-oss"
APPLY_FAILED = 97   # код возврата, если test_patch не применился (не путать с падением тестов)


def load_instances(data_dir, ids):
    from datasets import load_from_disk
    wanted, found = set(ids), {}
    for row in load_from_disk(data_dir)["train"]:
        ij = json.loads(row["extra_info"]["instance_json"])
        if ij["instance_id"] in wanted:
            found[ij["instance_id"]] = ij
    return found


def run_one(ij, pull, network, timeout):
    image = f"{REGISTRY}:{ij['instance_id']}"
    res = {"id": ij["instance_id"], "image": image}
    if pull:
        t0 = time.time()
        p = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
        res["pull_s"] = round(time.time() - t0, 1)
        if p.returncode:
            res["error"] = "pull failed: " + p.stderr.strip()[-300:]
            return res
    cwd = ij["cwd"]
    script = (f"cd {cwd} && (git apply --whitespace=nowarn - || exit {APPLY_FAILED}) && "
              f"bash mimo_test_command.sh")
    cmd = ["docker", "run", "--rm", "-i", "--memory", "4g", "--cpus", "2"]
    if not network:
        cmd += ["--network", "none"]
    cmd += [image, "bash", "-lc", script]
    t0 = time.time()
    try:
        # байты, а не text=True: на Windows текстовый режим превращает \n в \r\n и ломает патч
        p = subprocess.run(cmd, input=ij["test_patch"].encode("utf-8"), capture_output=True,
                           timeout=timeout)
        res["returncode"] = p.returncode
        out = (p.stdout + p.stderr).decode("utf-8", errors="replace")
        res["tail"] = out.replace("\r", "").strip().splitlines()[-12:]
    except subprocess.TimeoutExpired:
        res["error"] = f"timeout {timeout}s"
    res["run_s"] = round(time.time() - t0, 1)
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--network", action="store_true", help="разрешить сеть (по умолчанию выключена)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="pilot_report.json")
    args = ap.parse_args()

    found = load_instances(args.data, args.ids)
    results = []
    for tid in args.ids:
        if tid not in found:
            results.append({"id": tid, "error": "нет в датасете"})
            continue
        r = run_one(found[tid], args.pull, args.network, args.timeout)
        results.append(r)
        rc = r.get("returncode")
        verdict = ("ОШИБКА" if rc is None
                   else "ПАТЧ НЕ ПРИМЕНИЛСЯ (задачу не проверили)" if rc == APPLY_FAILED
                   else "ТЕСТЫ ПРОШЛИ ДО ИСПРАВЛЕНИЯ (подозрительно)" if rc == 0
                   else "ТЕСТЫ ПАДАЮТ (ожидаемо; проверьте хвост вывода)")
        print(f"{tid}: {verdict} rc={r.get('returncode')} {r.get('error', '')}")
        for line in r.get("tail", [])[-4:]:
            print("    ", line)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("->", args.out)


if __name__ == "__main__":
    main()
