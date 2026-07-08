# Flappy Bird RL — task runner.  Install `just` with:  brew install just
# Run `just` (no args) to list everything.
#
# The `py` path below adapts to the OS (mac/Linux use .venv/bin, Windows uses
# .venv/Scripts). The `bootstrap`/`venv` recipes assume a Unix shell (mac/Linux);
# on Windows create the venv manually:
#     python -m venv .venv && .venv\Scripts\pip install -e ".[dev,logging]"

# OS-aware venv python location
venv_py := if os_family() == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }
sys_py  := if os_family() == "windows" { "python" } else { "python3" }
# use the venv's python if it exists, else the system one
py := if path_exists(venv_py) == "true" { venv_py } else { sys_py }

# list all recipes
default:
    @just --list

# ---------------------------------------------------------------- setup

# create .venv and install the project + dev/logging extras (run this first)
bootstrap:
    #!/usr/bin/env bash
    set -euo pipefail
    {{sys_py}} -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -e ".[dev,logging]"
    echo ""
    echo "Bootstrapped. Activate with:  source .venv/bin/activate"

# create/ensure a venv (.venv, or .venvs/<name> if given) and install the project
venv name="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{name}}" ]; then dir=".venv"; else dir=".venvs/{{name}}"; fi
    if [ ! -d "$dir" ]; then
        echo "creating $dir ..."
        {{sys_py}} -m venv "$dir"
        "$dir/bin/pip" install --upgrade pip >/dev/null
        "$dir/bin/pip" install -e ".[dev,logging]"
    else
        echo "$dir already exists"
    fi
    echo ""
    echo "activate it:  source $dir/bin/activate"

# list the venvs you have
venvs:
    #!/usr/bin/env bash
    echo "default:"; [ -d .venv ] && echo "  .venv" || echo "  (none — run 'just bootstrap')"
    echo "named (.venvs/):"; ls -1 .venvs 2>/dev/null | sed 's/^/  /' || echo "  (none)"

# alias for `venvs`
view: venvs

# ---------------------------------------------------------------- DQN

# train and save the best snapshot -> models/<name>.pth  (default replaces dqn_final)
# examples:  just train        |  just train dqn2        |  just train dqn2 --steps 200000
train name="dqn_final" *opts:
    {{py}} -m flappy_rl.cli train {{name}} {{opts}}

# watch a model play:  just eval 5   |   just eval 6 dqn2.pth
eval *args:
    {{py}} -m flappy_rl.cli eval {{args}}

# quick windowed watch of the current model (5 games)
play:
    {{py}} -m flappy_rl.cli eval 5 --fps 45

# ---------------------------------------------------------------- other

# tabular Q-learning experiment (the non-DQN approach)
train-qlearn *args:
    {{py}} scripts/train_qlearn.py {{args}}

# run the test suite
test:
    {{py}} -m pytest -q

# delete generated training runs
clean:
    rm -rf experiments/*/ && echo "cleared experiments/"
