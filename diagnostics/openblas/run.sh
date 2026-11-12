#!/usr/bin/env bash
set -euo pipefail
phase=${1:?Select build or test}
: "${SAGE_SOURCE:?}"
: "${DIAGNOSTIC_OUT:?}"
: "${CONDA_PREFIX:?}"
helper=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/check.py
unset PYTHONPATH PYTHONHOME SAGE_ROOT SAGE_LOCAL SAGE_VENV SAGE_LIB SAGE_SRC SAGE_SHARE SAGE_DATA_PATH
# Minimal CI installs normally to ~/.local, so keep its normal user-site lookup.
unset PYTHONNOUSERSITE
export PYTHONDONTWRITEBYTECODE=1 PYDEVD_DISABLE_FILE_VALIDATION=1
cd -- "${SAGE_SOURCE}"

if [[ ${phase} == build ]]; then
  python "${helper}" environment
  meson setup --buildtype=release --python.install-env=prefix --prefix="${HOME}/.local" \
    --wrap-mode=nodownload -Dbuild-docs=false -Dbliss=disabled -Dcoxeter3=disabled \
    -Dlibbraiding=disabled -Dlibhomfly=disabled -Dmcqd=disabled -Dmeataxe=disabled \
    -Drankwidth=disabled -Dsirocco=disabled -Dtdlib=disabled builddir \
    2>&1 | tee "${DIAGNOSTIC_OUT}/meson-setup.log"
  meson install -C builddir 2>&1 | tee "${DIAGNOSTIC_OUT}/build.log"
  python -m pip check 2>&1 | tee "${DIAGNOSTIC_OUT}/pip-check.log"
  python "${helper}" installed
elif [[ ${phase} == test ]]; then
  overall=0
  # Each mode has fresh Python processes and separate startup caches. The first
  # preserves inherited BLAS settings; only the second sets the shell policy.
  modes=(baseline single-thread)
  if [[ ${DIAGNOSTIC_VARIANT} == candidate ]]; then modes=(candidate); fi
  for mode in "${modes[@]}"; do
    set +e
    (
      set -euo pipefail
      export DOT_SAGE="${DIAGNOSTIC_OUT}/cache-${mode}/dot-sage"
      export MPLCONFIGDIR="${DIAGNOSTIC_OUT}/cache-${mode}/matplotlib"
      export MAXIMA_OBJDIR="${DIAGNOSTIC_OUT}/cache-${mode}/maxima"
      export XDG_CACHE_HOME="${DIAGNOSTIC_OUT}/cache-${mode}/xdg"
      mkdir -p "${DOT_SAGE}" "${MPLCONFIGDIR}" "${MAXIMA_OBJDIR}" "${XDG_CACHE_HOME}"
      if [[ ${mode} == single-thread ]]; then export OPENBLAS_NUM_THREADS=1; fi
      # Prove the real candidate entrypoint overrides a conflicting inherited value.
      if [[ ${mode} == candidate ]]; then export OPENBLAS_NUM_THREADS=4; fi
      # Probes run in separate processes; they cannot initialize the doctest's BLAS.
      set +e
      timeout --signal=TERM --kill-after=10s 120s python "${helper}" probe "${mode}" \
        > "${DIAGNOSTIC_OUT}/probe-${mode}.log" 2>&1
      probe_status=$?
      set -e
      printf '%s\n' "${probe_status}" > "${DIAGNOSTIC_OUT}/probe-exit-${mode}.txt"
      command=(python3 -m sage.doctest -p4 --format github --warn-long 5.0
        --random-seed=35990669349480983660881921148984004251
        "--stats-path=${DIAGNOSTIC_OUT}/timings-${mode}.json"
        "--logfile=${DIAGNOSTIC_OUT}/doctest-${mode}.log"
        src/sage/matrix/matrix_integer_dense.pyx)
      printf '%q ' "${command[@]}" > "${DIAGNOSTIC_OUT}/command-${mode}.txt"
      printf '\n' >> "${DIAGNOSTIC_OUT}/command-${mode}.txt"
      "${command[@]}" 2>&1 | tee "${DIAGNOSTIC_OUT}/console-${mode}.log"
      exit "${probe_status}"
    )
    status=$?
    set -e
    printf '%s\n' "${status}" > "${DIAGNOSTIC_OUT}/exit-${mode}.txt"
    if (( status != 0 )); then overall=1; fi
  done
  python "${helper}" final || overall=1
  printf '%s\n' "${overall}" > "${DIAGNOSTIC_OUT}/overall-exit.txt"
  exit "${overall}"
else
  printf 'Unknown phase: %s\n' "${phase}" >&2
  exit 2
fi
