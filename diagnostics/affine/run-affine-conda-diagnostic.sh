#!/usr/bin/env bash
# Staged locally; intended location diagnostics/affine/ on the diagnostic branch.
set -euo pipefail

phase=${1:?Select build or test}
: "${SAGE_SOURCE:?Set the immutable upstream source checkout}"
: "${DIAGNOSTIC_OUT:?Set a new artifact directory}"
: "${CONDA_PREFIX:?Activate the pinned sage-dev Conda environment}"
harness_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
helper="${harness_dir}/affine-conda-diagnostic.py"
[[ -d ${DIAGNOSTIC_OUT} ]] || { printf 'Run source preflight first.\n' >&2; exit 2; }

# Keep source/import isolation explicit; preserve inherited numerical thread settings.
unset PYTHONPATH PYTHONHOME SAGE_ROOT SAGE_LOCAL SAGE_VENV SAGE_LIB SAGE_SRC SAGE_SHARE SAGE_DATA_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export DOT_SAGE="${DIAGNOSTIC_OUT}/cache/dot-sage"
export MPLCONFIGDIR="${DIAGNOSTIC_OUT}/cache/matplotlib"
export MAXIMA_OBJDIR="${DIAGNOSTIC_OUT}/cache/maxima"
export XDG_CACHE_HOME="${DIAGNOSTIC_OUT}/cache/xdg"
mkdir -p "${DOT_SAGE}" "${MPLCONFIGDIR}" "${MAXIMA_OBJDIR}" "${XDG_CACHE_HOME}"
cd -- "${SAGE_SOURCE}"

if [[ ${phase} == build ]]; then
  python "${helper}" environment
  # Same non-editable build command as the original standard Conda CI.
  # This command does not update Python dependencies or apply CI-fix patches.
  python -m pip install --no-build-isolation --no-deps \
    --config-settings=builddir=builddir . -v 2>&1 | tee "${DIAGNOSTIC_OUT}/build.log"
  python -m pip check 2>&1 | tee "${DIAGNOSTIC_OUT}/pip-check.log"
  python "${helper}" installed
elif [[ ${phase} == test ]]; then
  python "${helper}" installed
  overall=0
  seeds=(88850114243507317388609685677107657968 119624209447655956655053293230659760303)
  # These are the two original Python3.14 seeds, not replacements for failures.
  for seed in "${seeds[@]}"; do
    command=(./sage -t -p4 --format github --warn-long 5.0
      "--random-seed=${seed}" "--stats-path=${DIAGNOSTIC_OUT}/timings-${seed}.json"
      "--logfile=${DIAGNOSTIC_OUT}/doctest-${seed}.log"
      src/sage/schemes/affine/affine_homset.py)
    printf '%q ' "${command[@]}" > "${DIAGNOSTIC_OUT}/command-${seed}.txt"
    printf '\n' >> "${DIAGNOSTIC_OUT}/command-${seed}.txt"
    set +e
    "${command[@]}" 2>&1 | tee "${DIAGNOSTIC_OUT}/console-${seed}.log"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    printf '%s\n' "${pipeline_status[0]}" > "${DIAGNOSTIC_OUT}/exit-${seed}.txt"
    if (( pipeline_status[0] != 0 || pipeline_status[1] != 0 )); then overall=1; fi
  done

  # Fresh processes, after both untouched-file tests; failures remain failures.
  for seed in "${seeds[@]}"; do
    set +e
    python "${helper}" instrument --seed "${seed}" \
      2>&1 | tee "${DIAGNOSTIC_OUT}/instrument-${seed}.log"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    printf '%s\n' "${pipeline_status[0]}" > "${DIAGNOSTIC_OUT}/instrument-exit-${seed}.txt"
    if (( pipeline_status[0] != 0 || pipeline_status[1] != 0 )); then overall=1; fi
  done
  python "${helper}" verify-after || overall=1
  printf '%s\n' "${overall}" > "${DIAGNOSTIC_OUT}/overall-exit.txt"
  exit "${overall}"
else
  printf 'Unknown phase: %s\n' "${phase}" >&2
  exit 2
fi
