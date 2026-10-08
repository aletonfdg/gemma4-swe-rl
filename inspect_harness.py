"""
Проверяем, что реально импортируется из коробки в Kaggle-окружении.
ЗАПУСТИТЬ ПЕРВЫМ, ДО ЛЮБОЙ РАБОТЫ С sim/.

Если swegemma.harness.verification.verify_task (или аналог) импортируется -
НЕМЕДЛЕННО переключить sim/verifier.py и sim/swe_env.py на прямой вызов
оригинальных функций харнесса вместо нашей реконструкции. Это убирает весь
риск расхождения в Phase 2 verification / anti-tampering логике.
"""
import importlib

CANDIDATES = [
    "swegemma",
    "swegemma.tools",
    "swegemma.context",
    "swegemma.harness",
    "swegemma.harness.verification",
    "swegemma.harness.container_setup",
    "swegemma.scripts.inference",
    "swegemma.scripts.metric",
    "swegemma.prompts",
    "adk_eval_core",
    "adk_submission",
]

if __name__ == "__main__":
    for name in CANDIDATES:
        try:
            mod = importlib.import_module(name)
            print(f"[OK]   {name} -> {getattr(mod, '__file__', '?')}")
            public = [n for n in dir(mod) if not n.startswith("_")]
            print("       exports:", public[:30])
        except Exception as e:
            print(f"[MISS] {name}: {e}")
