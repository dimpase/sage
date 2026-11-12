#!/usr/bin/env python3
"""Standalone evidence harness; never edits Sage source or installs packages."""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

BASE = "026e931e47064f0a7fc095580119365518bfe7c0"
PIN_FILE = "environment-3.14-linux.yml"
EXPECTED_HASHES = {
    PIN_FILE: "adf46aad5de7417f5c29f6d58b9cf29a8044879ba4d357a7889d74a771c33ae8",
    "src/sage/schemes/affine/affine_homset.py": "d9e5e7f6c659dd7ff5096d41076fcce8a9a3f2186fa4147ab9c0aa4d205643cd",
    "src/sage/rings/polynomial/polynomial_element.pyx": "87cd78754ac8809691231049477ac0a3b27b226ab2d4fd809babc3f198ba61a6",
    "src/sage/rings/complex_double.pyx": "1e701e183d8aa2cf3d3f763cf885976611a636578786494530da84816f83e36f",
    "src/sage/rings/polynomial/multi_polynomial_ideal.py": "3aee3a8119c11f72c028e08c480e1f8cf971b041a7cbb3a939dc505ca1b75e19",
}
SEEDS = (
    "88850114243507317388609685677107657968",
    "119624209447655956655053293230659760303",
)
ENV_KEYS = (
    "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE",
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "SAGE_NUM_THREADS",
    "OPENBLAS_CORETYPE", "LD_LIBRARY_PATH", "LD_PRELOAD",
    "DOT_SAGE", "MPLCONFIGDIR", "MAXIMA_OBJDIR", "XDG_CACHE_HOME",
    "RUNNER_OS", "RUNNER_ARCH", "ImageOS", "ImageVersion",
    "GITHUB_REPOSITORY", "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(out, name, value):
    (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(argv, cwd=None):
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    return {"argv": argv, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def checked(argv, cwd=None):
    result = run(argv, cwd)
    require(result["returncode"] == 0, repr(result))
    return result["stdout"]


def check_source(source):
    head = checked(["git", "rev-parse", "HEAD"], source).strip()
    require(head == BASE, f"Source HEAD changed: {head}")
    diff = checked(["git", "diff", "--name-only", "HEAD"], source)
    require(not diff, f"Tracked upstream files changed: {diff}")
    hashes = {name: sha(source / name) for name in EXPECTED_HASHES}
    require(hashes == EXPECTED_HASHES, f"Unexpected source bytes: {hashes}")
    return {"head": head, "hashes": hashes,
            "git_status": checked(["git", "status", "--short"], source)}


def pinned_packages(text):
    require("dependencies:\n" in text, "Missing dependencies section")
    rows = text.split("dependencies:\n", 1)[1].splitlines()
    result = {}
    for line in rows:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        require(line.startswith("  - "), f"Unsupported lock entry: {line}")
        parts = line[4:].split("=")
        require(len(parts) == 3 and all(parts), f"Unpinned lock entry: {line}")
        name, version, build = parts
        require(name not in result, f"Duplicate pin: {name}")
        result[name] = {"version": version, "build": build}
    return result


def pin_errors(expected, actual):
    by_name = {row["name"]: row for row in actual}
    errors = []
    for name, pin in expected.items():
        row = by_name.get(name)
        observed = None if row is None else {
            "version": row["version"],
            "build": row.get("build_string", row.get("build")),
        }
        if observed != pin:
            errors.append({"name": name, "expected": pin, "actual": observed})
    return errors


def source_phase(source, out):
    # A run/attempt owns its artifacts. Refuse to overwrite an earlier attempt.
    out.mkdir(parents=True, exist_ok=False)
    metadata = check_source(source)
    metadata["scope"] = "Pristine upstream beta6 baseline; no PR or CI-fix patches"
    metadata["environment"] = {key: os.environ.get(key) for key in ENV_KEYS}
    metadata["machine"] = platform.machine()
    metadata["platform"] = platform.platform()
    metadata["os_release"] = Path("/etc/os-release").read_text()
    metadata["cpu"] = run(["lscpu"])
    save(out, "source.json", metadata)
    require(platform.machine() == "x86_64", "This is not the required x86_64 baseline")
    require('VERSION_ID="26.04"' in metadata["os_release"], "Expected Ubuntu26.04")
    here = Path(__file__).resolve().parent
    harness_root = here.parents[1]
    copies = [here / "affine-conda-diagnostic.py", here / "run-affine-conda-diagnostic.sh",
              harness_root / ".github/workflows/affine-conda-diagnostic.yml"]
    target = out / "harness"
    target.mkdir()
    for path in copies:
        shutil.copyfile(path, target / path.name)
    save(out, "harness-hashes.json", {path.name: sha(path) for path in copies})


def environment_phase(source, out, name="environment.json"):
    require(Path(sys.prefix).resolve() == Path(os.environ["CONDA_PREFIX"]).resolve(),
            "Active Python is not in the intended Conda environment")
    expected = pinned_packages((source / PIN_FILE).read_text())
    actual = json.loads(checked(["conda", "list", "--json"]))
    # conda list can label an installed Conda distribution pypi_0 (the original
    # logs do this for rpy2). Check the actual Conda package receipts instead.
    records = [json.loads(path.read_text())
               for path in sorted((Path(sys.prefix) / "conda-meta").glob("*.json"))]
    errors = pin_errors(expected, records)
    metadata = {"python": sys.version, "executable": sys.executable,
                "prefix": sys.prefix, "machine": platform.machine(),
                "pinned_count": len(expected), "packages": actual,
                "conda_package_records": [
                    {key: row.get(key) for key in (
                        "name", "version", "build", "build_number", "channel",
                        "subdir", "url", "sha256", "md5", "depends")}
                    for row in records],
                "pin_errors": errors,
                "environment": {key: os.environ.get(key) for key in ENV_KEYS},
                "conda_info": run(["conda", "info"])}
    save(out, name, metadata)
    if name == "environment.json":
        (out / "conda-explicit.txt").write_text(checked(["conda", "list", "--explicit"]))
        shutil.copyfile(source / PIN_FILE, out / PIN_FILE)
    require(not errors, f"Conda environment differs from exact original pins: {errors}")
    require(platform.machine() == "x86_64", "Wrong architecture")
    return metadata


def installed_phase(source, out):
    import importlib
    import sage.all
    import sage.version
    import numpy
    import scipy
    from sage.features.sagemath import sage__libs__singular

    prefix = Path(sys.prefix).resolve()
    require(numpy.__version__ == "2.5.4", "Unexpected imported NumPy version")
    require(scipy.__version__ == "1.18.1", "Unexpected imported SciPy version")
    modules = {}
    for name in ("sage.all", "sage.schemes.affine.affine_homset",
                 "sage.rings.polynomial.polynomial_element", "sage.rings.complex_double",
                 "sage.libs.singular.singular", "numpy", "scipy"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(prefix), f"Module escaped Conda prefix: {name}: {path}")
        modules[name] = {"path": str(path), "sha256": sha(path)}
    affine = modules["sage.schemes.affine.affine_homset"]["sha256"]
    require(affine == EXPECTED_HASHES["src/sage/schemes/affine/affine_homset.py"],
            "Installed affine module is not the unchanged baseline")
    sage__libs__singular().require()
    result = {"source": check_source(source), "python": sys.version,
              "sage_version": sage.version.version, "numpy": numpy.__version__,
              "scipy": scipy.__version__, "modules": modules}
    save(out, "installed.json", result)
    return result


def binary64(value):
    value = float(value)
    return {"decimal": value if math.isfinite(value) else str(value), "hex": value.hex()}


def complex_record(value):
    value = complex(value)
    return {"real": binary64(value.real), "imag": binary64(value.imag)}


def numpy_runtime(numpy):
    result = {"environment": {key: os.environ.get(key) for key in ENV_KEYS}}
    for name in ("show_config", "show_runtime"):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            getattr(numpy, name)()
        result[name] = output.getvalue()
    try:
        import threadpoolctl
    except ImportError:
        result["threadpoolctl"] = "Not installed; no package added"
    else:
        result["threadpoolctl"] = threadpoolctl.threadpool_info()
    # Dynamic-loader evidence for the current process, not just package metadata.
    maps = Path("/proc/self/maps").read_text().splitlines()
    result["loaded_numeric_library_maps"] = [
        line for line in maps
        if any(word in line.lower() for word in
               ("blas", "lapack", "_umath_linalg", "singular", "libgfortran", "libgomp"))
    ]
    return result


def instrument(source, out, seed):
    from copy import copy
    import numpy
    from sage.all import AffineSpace, CDF, ComplexField, PolynomialRing, QQ, RR, set_random_seed
    from sage.structure.element import get_coercion_model

    require(seed in SEEDS, "Only the two recorded Python3.14 seeds are accepted")
    source_before = check_source(source)
    result = {"status": "started", "source": source_before, "seed": seed,
              "numpy_calls": [], "stages": [], "candidates": []}
    numpy_roots = numpy.roots
    active_call = {}

    def recording_roots(coefficients):
        require(str(coefficients.dtype) == "complex128",
                f"Unexpected root coefficient dtype: {coefficients.dtype}")
        entry = {"context": dict(active_call), "dtype": str(coefficients.dtype),
                 "coefficients_descending": [complex_record(c) for c in coefficients]}
        result["numpy_calls"].append(entry)
        try:
            roots = numpy_roots(coefficients)
        except Exception as error:
            entry["exception"] = repr(error)
            raise
        entry["returned_count"] = len(roots)
        entry["returned_roots"] = [complex_record(r) for r in roots]
        return roots  # Return the unchanged NumPy array, not a reconstructed copy.

    def coordinates(point):
        return {str(variable): complex_record(value) for variable, value in point.items()}

    try:
        set_random_seed(int(seed))
        ambient = AffineSpace(QQ, 2, names=("x1", "x2"))
        x1, x2 = ambient.gens()
        scheme = ambient.subscheme([30*x1**100 + 1000*x2**2 + 2000*x1*x2 + 1, x1+x2])
        # A direct untouched method call is a control, before instrumenting NumPy.
        control = scheme(QQ).numerical_points(F=CDF, zero_tolerance=1e-9)
        result["untouched_method_count"] = len(control)
        result["untouched_method_points"] = [[complex_record(v) for v in p] for p in control]

        set_random_seed(int(seed))
        numerical_ambient = ambient.change_ring(CDF)
        dimension = scheme.defining_ideal().dimension()
        require(dimension == 0, f"Unexpected ideal dimension: {dimension}")
        tolerance = RR(1e-9)
        ring = PolynomialRing(QQ, 2, ambient.gens(), order="lex")
        floating_ring = ring.change_ring(CDF)
        ideal = ring.ideal(scheme.defining_polynomials())
        exact_basis = ideal.groebner_basis()
        basis = [floating_ring(g) for g in exact_basis]
        result["exact_basis"] = [str(g) for g in exact_basis]
        result["floating_basis"] = [str(g) for g in basis]
        result["polynomial_classes"] = [type(g).__module__ + "." + type(g).__name__ for g in basis]
        result["tolerance"] = {"repr": repr(tolerance), "binary64": binary64(tolerance)}
        require(basis != [1], "Unexpected empty scheme")

        numpy.roots = recording_roots
        points = [{}]
        # Mirror the production traversal and mutations; add observation only.
        for stage_index, polynomial in enumerate(reversed(basis)):
            stage = {"polynomial": str(polynomial), "input_count": len(points), "roots": []}
            new_points = []
            good = False
            for point_index, point in enumerate(points):
                substituted = polynomial.substitute(point)
                if len(floating_ring(substituted).variables()) == 1:
                    variable = substituted.variables()[0]
                    variable = floating_ring.gen(floating_ring.gens().index(variable))
                    univariate = substituted.univariate_polynomial()
                    active_call.clear()
                    active_call.update(stage=stage_index, point=point_index)
                    call_start = len(result["numpy_calls"])
                    roots = univariate.roots(ring=CDF, multiplicities=False)
                    stage["roots"].append({
                        "point_index": point_index, "degree": int(univariate.degree()),
                        "numpy_call_indices": list(range(call_start, len(result["numpy_calls"]))),
                        "sage_roots_after_exact_dedup_count": len(roots),
                        "sage_roots": [complex_record(r) for r in roots],
                    })
                    for root in roots:
                        point[variable] = root
                        new_points.append(copy(point))
                        good = True
                else:
                    new_points.append(point)
                    good = True
            if good:
                points = new_points
            stage["output_count"] = len(points)
            stage["coordinate_dictionaries"] = [coordinates(p) for p in points]
            result["stages"].append(stage)
        numpy.roots = numpy_roots

        defining_polynomials = [g.change_ring(CDF) for g in scheme.defining_polynomials()]
        accepted_points = []
        high_precision = ComplexField(200)
        coercion_model = get_coercion_model()
        for index, point in enumerate(points):
            record = {"index": index, "coordinates": coordinates(point), "complete": len(point) == 2}
            result["candidates"].append(record)
            if len(point) != 2:
                continue
            candidate = numerical_ambient([point[ring.gen(j)] for j in range(2)])
            # This is the original predicate and evaluation, with short-circuiting.
            accepted = all(g(list(candidate)) < tolerance for g in defining_polynomials)
            if accepted:
                accepted_points.append(candidate)
            residual_records = []
            for polynomial in defining_polynomials:
                residual = polynomial(list(candidate))
                left, right = coercion_model.canonical_coercion(residual, tolerance)
                residual_records.append({
                    "value": complex_record(residual), "magnitude": binary64(abs(residual)),
                    "original_predicate": bool(residual < tolerance),
                    "magnitude_predicate_diagnostic_only": bool(abs(residual) < tolerance),
                    "coerced_parent": str(left.parent()),
                    "coerced_threshold": str(right),
                })
            a, b = [high_precision(complex(v)) for v in candidate]
            high_residuals = [30*a**100 + 1000*b**2 + 2000*a*b + 1, a+b]
            record.update(accepted=bool(accepted), residuals=residual_records,
                          residuals_200bit_same_binary64_candidates=[
                              {"real": str(z.real()), "imag": str(z.imag()), "abs": str(abs(z))}
                              for z in high_residuals])
        accepted_points = sorted(accepted_points)
        result["candidate_count"] = len(points)
        result["complete_count"] = sum(c["complete"] for c in result["candidates"])
        result["accepted_count"] = len(accepted_points)
        result["rejected_count"] = sum(c.get("accepted") is False for c in result["candidates"])
        result["exact_points_equal_to_untouched_method"] = accepted_points == control
        result["runtime"] = numpy_runtime(numpy)
        result["source_after"] = check_source(source)
        require(accepted_points == control,
                "Instrumented points differ from the untouched control; inspect nondeterminism/context")
        result["status"] = "instrumentation_completed"
        print(json.dumps({key: result[key] for key in (
            "status", "seed", "untouched_method_count", "candidate_count",
            "complete_count", "accepted_count", "rejected_count")}, indent=2))
    except BaseException as error:
        result["status"] = "instrumentation_error"
        result["error"] = repr(error)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        numpy.roots = numpy_roots
        save(out, f"instrument-{seed}.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("source", "environment", "installed", "instrument", "verify-after"))
    parser.add_argument("--seed", choices=SEEDS)
    args = parser.parse_args()
    source = Path(os.environ["SAGE_SOURCE"]).resolve()
    out = Path(os.environ["DIAGNOSTIC_OUT"]).resolve()
    if args.phase == "source":
        source_phase(source, out)
    elif args.phase == "environment":
        environment_phase(source, out)
    elif args.phase == "installed":
        installed_phase(source, out)
    elif args.phase == "instrument":
        require(args.seed is not None, "instrument requires an original seed")
        instrument(source, out, args.seed)
    else:
        save(out, "source-after.json", check_source(source))
        environment_phase(source, out, "environment-after.json")
        installed_phase(source, out)


if __name__ == "__main__":
    main()
