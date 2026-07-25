#!/usr/bin/env bash
# Source this file: source scripts/activate_conda_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "error: source this script instead of executing it:" >&2
  echo "  source scripts/activate_conda_env.sh" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT_DIR/conda-env"
ENV_BIN="$ENV_DIR/bin"

if [[ ! -d "$ENV_BIN" ]]; then
  echo "error: missing environment at $ENV_DIR" >&2
  echo "create it first, then re-run this command" >&2
  return 1
fi

if [[ -n "${PROJECT_CONDA_ENV_ACTIVE:-}" ]]; then
  echo "project conda env is already active: $PROJECT_CONDA_ENV_ACTIVE"
  if [[ "${PS1-}" != "(conda-env) "* ]]; then
    export PS1="(conda-env) ${PS1-}"
  fi
  return 0
fi

export _PROJECT_CONDA_OLD_PATH="$PATH"
export _PROJECT_CONDA_OLD_PS1="${PS1-}"
export PATH="$ENV_BIN:$PATH"
export CONDA_PREFIX="$ENV_DIR"
export CONDA_DEFAULT_ENV="$ENV_DIR"
export PROJECT_CONDA_ENV_ACTIVE="$ENV_DIR"
export PS1="(conda-env) ${PS1-}"

deactivate_project_conda_env() {
  if [[ -n "${_PROJECT_CONDA_OLD_PATH:-}" ]]; then
    export PATH="$_PROJECT_CONDA_OLD_PATH"
    unset _PROJECT_CONDA_OLD_PATH
  fi
  if [[ -n "${_PROJECT_CONDA_OLD_PS1+x}" ]]; then
    export PS1="$_PROJECT_CONDA_OLD_PS1"
    unset _PROJECT_CONDA_OLD_PS1
  fi
  unset CONDA_PREFIX
  unset CONDA_DEFAULT_ENV
  unset PROJECT_CONDA_ENV_ACTIVE
  unset -f deactivate_project_conda_env
}

echo "Activated project conda env: $ENV_DIR"
echo "Run: datalad --version"
echo "Deactivate with: deactivate_project_conda_env"
