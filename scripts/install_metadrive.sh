#!/usr/bin/env bash
set -euo pipefail

metadrive_repo="https://github.com/HenryLHH/metadrive_clean.git"
metadrive_commit="ebcb48c44a74466d10cd01217f8b64b7763ff6c1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON:-python}"

if ! "${python_bin}" -c "import gym" >/dev/null 2>&1; then
    echo "Gym is not available in ${python_bin}. Activate fusion-baselines-v1 first." >&2
    exit 1
fi

"${python_bin}" - <<'PY'
import gym
import sys

if gym.__version__ != "0.21.0":
    raise SystemExit(
        "Expected gym==0.21.0 before installing MetaDrive; found "
        + gym.__version__
    )
print("Python:", sys.executable)
print("Gym:", gym.__version__)
PY

metadrive_tmp="$(mktemp -d /tmp/fusion-metadrive.XXXXXX)"
trap 'rm -rf -- "${metadrive_tmp}"' EXIT

git init --quiet "${metadrive_tmp}"
git -C "${metadrive_tmp}" remote add origin "${metadrive_repo}"
git -C "${metadrive_tmp}" fetch --depth 1 origin "${metadrive_commit}"
git -C "${metadrive_tmp}" checkout --quiet --detach FETCH_HEAD

actual_commit="$(git -C "${metadrive_tmp}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${metadrive_commit}" ]]; then
    echo "MetaDrive commit mismatch: ${actual_commit}" >&2
    exit 1
fi

git -C "${metadrive_tmp}" apply \
    "${project_dir}/patches/metadrive-gym021.patch"

"${python_bin}" -m pip install --no-deps "${metadrive_tmp}"
"${python_bin}" -m pip check

echo "Installed MetaDrive ${metadrive_commit} with Gym 0.21 metadata."
