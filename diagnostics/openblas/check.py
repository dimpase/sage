"""Read-only provenance and separate-process OpenBLAS startup observations."""

import ctypes
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE = "026e931e47064f0a7fc095580119365518bfe7c0"
HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ["SAGE_SOURCE"]).resolve()
OUT = Path(os.environ["DIAGNOSTIC_OUT"])
MANIFEST = json.loads((HERE / "manifest.json").read_text())
VARIANT = os.environ["DIAGNOSTIC_VARIANT"]
EXPECTED = MANIFEST["variants"][VARIANT]
ENV_KEYS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "GOTO_NUM_THREADS",
    "OPENBLAS_CORETYPE", "SAGE_NUM_THREADS", "SAGE_TIMEOUT", "SAGE_TIMEOUT_LONG",
    "PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "CONDA_PREFIX",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "RUNNER_OS", "RUNNER_ARCH", "ImageOS",
    "ImageVersion", "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT",
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def command(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True)


def source():
    require(command(["git", "rev-parse", "HEAD"], SOURCE).strip() == EXPECTED["commit"], "Wrong source")
    require(not command(["git", "diff", "HEAD", "--name-only"], SOURCE), "Modified source")
    hashes = {name: sha(SOURCE / name) for name in EXPECTED["source_hashes"]}
    require(hashes == EXPECTED["source_hashes"], "Source bytes differ")
    return {"head": EXPECTED["commit"], "variant": VARIANT, "hashes": hashes,
            "status": command(["git", "status", "--short"], SOURCE)}


def environment(name):
    prefix = Path(sys.prefix).resolve()
    require(prefix == Path(os.environ["CONDA_PREFIX"]).resolve(), "Wrong Python prefix")
    records = [json.loads(p.read_text()) for p in sorted((prefix / "conda-meta").glob("*.json"))]
    actual = {r["name"]: [r["version"], r["build"]] for r in records}
    expected = MANIFEST["conda_receipts"]
    differences = {n: {"expected": expected.get(n), "actual": actual.get(n)}
                   for n in set(actual) | set(expected) if actual.get(n) != expected.get(n)}
    result = {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
              "records": records, "differences": differences,
              "environment": {key: os.environ.get(key) for key in ENV_KEYS}}
    save(name, result)
    require(not differences, f"Conda receipt mismatch: {differences}")
    require(platform.machine() == "x86_64", "Wrong architecture")
    return result


def installed():
    import sage.all  # noqa: F401

    modules = {}
    prefix = (Path.home() / ".local").resolve()
    for name in ("sage.all", "sage.doctest.__main__", "sage.matrix.matrix_integer_dense"):
        path = Path(importlib.import_module(name).__file__).resolve()
        require(path.is_relative_to(prefix), f"Unexpected import: {name}: {path}")
        modules[name] = {"path": str(path), "sha256": sha(path)}
    require(modules["sage.doctest.__main__"]["sha256"] ==
            EXPECTED["source_hashes"]["src/sage/doctest/__main__.py"], "Changed entrypoint")
    save("installed.json", {"source": source(), "modules": modules})


def mapped_blas():
    return sorted({line.split(maxsplit=5)[5] for line in Path("/proc/self/maps").read_text().splitlines()
                   if "libopenblas" in line and len(line.split(maxsplit=5)) == 6})


def probe(mode):
    observations = {"initial": mapped_blas()}
    initial_policy = os.environ.get("OPENBLAS_NUM_THREADS")
    for name in ("sage.doctest", "sage.doctest.__main__"):
        importlib.import_module(name)
        observations[name] = mapped_blas()
        save(f"probe-imports-{mode}.json", observations)
    require(os.environ.get("OPENBLAS_NUM_THREADS") == initial_policy,
            "Plain module import changed OpenBLAS policy")
    # Run the real entrypoint through its imports, stopping at argparse --help.
    # No monkeypatch, alternate initializer, or injected source is used.
    from sage.doctest.__main__ import main

    sys.argv = ["sage.doctest", "--help"]
    try:
        main()
    except SystemExit as exc:
        require(exc.code == 0, "--help failed")
    else:
        raise RuntimeError("Expected --help to exit")
    observations["after_main_help"] = mapped_blas()
    save(f"probe-imports-{mode}.json", observations)
    # Match the current Linux main() policy before init_sage imports Sage's maths.
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["SAGE_NUM_THREADS"] = "2"
    from sage.doctest.forker import init_sage

    init_sage()
    maps = mapped_blas()
    observations["init_sage"] = maps
    save(f"probe-imports-{mode}.json", observations)
    require(len(maps) == 1, f"Expected one OpenBLAS: {maps}")
    path = Path(maps[0]).resolve(strict=True)
    require(path.is_relative_to(Path(sys.prefix)), "OpenBLAS outside Conda")
    library = ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD | os.RTLD_LOCAL | os.RTLD_NOW)
    getters = {}
    for name, restype in (("openblas_get_num_threads", ctypes.c_int),
                          ("openblas_get_parallel", ctypes.c_int),
                          ("openblas_get_config", ctypes.c_char_p),
                          ("openblas_get_corename", ctypes.c_char_p)):
        fn = getattr(library, name)
        fn.argtypes = []
        fn.restype = restype
        value = fn()
        getters[name] = value.decode() if isinstance(value, bytes) else value
    result = {"mode": mode, "variant": VARIANT, "initial_policy": initial_policy,
              "observations": observations, "path": str(path),
              "sha256": sha(path), "getters": getters,
              "configured_threads_are_not_active_worker_counts": True,
              "environment": {key: os.environ.get(key) for key in ENV_KEYS}}
    save(f"probe-{mode}.json", result)
    if mode == "single-thread" or VARIANT == "candidate":
        require(getters["openblas_get_num_threads"] == 1, "Policy not effective")


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "source":
        OUT.mkdir(parents=True, exist_ok=False)
        require(platform.machine() == "x86_64", "Expected x86_64")
        require('VERSION_ID="26.04"' in Path("/etc/os-release").read_text(), "Wrong OS")
        save("source.json", source())
        save("host.json", {"cpu": command(["lscpu"]), "platform": platform.platform(),
                           "environment": {key: os.environ.get(key) for key in ENV_KEYS}})
        shutil.copytree(HERE, OUT / "harness")
        shutil.copyfile(HERE.parents[1] / ".github/workflows/openblas-fork-diagnostic.yml",
                        OUT / "harness/workflow.yml")
    elif phase == "environment":
        environment("environment.json")
        (OUT / "conda-explicit.txt").write_text(command(["conda", "list", "--explicit"]))
    elif phase == "installed":
        installed()
    elif phase == "probe":
        probe(sys.argv[2])
    elif phase == "final":
        environment("environment-after.json")
        save("source-after.json", source())
    else:
        raise ValueError(phase)
