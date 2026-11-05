#!/bin/bash
# Проверка skills/tmp-bridge в Linux-контейнере:
#   docker run --rm -v <repo>/skills/tmp-bridge:/skill:ro -v <repo>/tests:/t:ro swebench-sandbox bash /t/check_tmp_bridge.sh
#  1) Docker-песочница: репозиторий /workspace, настоящий tmp -- /tmp;
#  2) подпроцессная (Kaggle): /tmp/swegemma_sandbox_X/{workspace,tmp}, скрипт запускается из копии навыка,
#     исходный cwd лежит в локальной переменной _orig_cwd (как у обёртки ADK).
# write_file("/tmp/r.py") эмулируем записью в <repo>/tmp/r.py, run_command -- чтением из настоящего tmp;
# git add -N . && git diff HEAD (как submit_patch) не должен содержать tmp/ и repro.py.
set -e
check() {  # $1 repo, $2 real tmp
  local repo=$1 rtmp=$2
  echo 'print("NEW")' > "$repo/tmp/r.py"
  [ "$(cat "$rtmp/r.py")" = 'print("NEW")' ] || { echo "FAIL: real tmp does not see the file"; exit 1; }
  echo 'x = 1' > "$repo/repro.py"
  echo '# fix' >> "$repo/src.py"
  (cd "$repo" && git add -N . && git diff HEAD --name-only) > /tmp/names.txt
  grep -qx 'src.py' /tmp/names.txt || { echo "FAIL: real change missing"; exit 1; }
  if grep -Eq '^(tmp|repro.py)' /tmp/names.txt; then echo "FAIL: scratch in patch"; cat /tmp/names.txt; exit 1; fi
  echo "OK: $(tr '\n' ' ' < /tmp/names.txt)"
}
mkrepo() {
  mkdir -p "$1" && cd "$1" && git init -q && git config user.email t@t && git config user.name t
  echo 'a = 1' > src.py && git add . && git commit -qm init && cd - > /dev/null
}

# 1) Docker layout
mkrepo /workspace
cd /skill && python3 scripts/setup.py
python3 scripts/setup.py
check /workspace /tmp
rm -rf /workspace

# 2) subprocess layout
SB=/tmp/swegemma_sandbox_1_; mkdir -p $SB/tmp; mkrepo $SB/workspace
COPY=$(mktemp -d); cp -r /skill/scripts "$COPY/"
cd $SB/workspace && python3 -c "
import os, runpy
_orig_cwd = os.getcwd()
os.chdir('$COPY')
runpy.run_path('scripts/setup.py', run_name='__main__')
"
check $SB/workspace $SB/tmp
echo ALL OK
