"""Checks shared by the Homebrew dependency bottle workflow phases."""

import importlib
from importlib import metadata
from importlib.resources import files
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib


def requirements():
    """Use the candidate's locked versions for the two pytest entry points."""
    with open("uv.lock", "rb") as stream:
        packages = tomllib.load(stream)["package"]
    for name in ("pytest", "pytest-xdist"):
        versions = {p["version"] for p in packages if p["name"] == name}
        if len(versions) != 1:
            raise RuntimeError(f"Expected one locked version of {name}: {versions}")
        print(f"{name}=={versions.pop()}")


def meson():
    """Expose the genuine pure-Python Meson bottle to this development venv."""
    from email.parser import Parser
    import platform
    import shutil
    import sysconfig

    from packaging.specifiers import SpecifierSet

    venv = Path(os.environ["CI_RUN_DIR"], "venv").resolve(strict=True)
    purelib = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    if (sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != venv
            or not purelib.is_relative_to(venv)):
        raise RuntimeError("Meson must be exposed only inside the CI development venv")
    prefix = Path(subprocess.check_output(
        ["brew", "--prefix", "meson"], text=True).strip()).resolve(strict=True)
    receipt = json.loads((prefix / "INSTALL_RECEIPT.json").read_text())
    if (receipt.get("source", {}).get("tap") != "homebrew/core"
            or receipt.get("poured_from_bottle") is not True):
        raise RuntimeError(f"Expected a poured Homebrew core Meson bottle: {prefix}")
    locations = list(prefix.glob("lib/python*/site-packages/meson-*.dist-info"))
    if len(locations) != 1:
        raise RuntimeError(f"Expected one bottled Meson distribution: {locations}")
    info = locations[0].resolve(strict=True)
    site = info.parent
    if (not site.is_relative_to(prefix)
            or {path.name for path in site.iterdir()} != {info.name, "mesonbuild"}
            or not (site / "mesonbuild").resolve(strict=True).is_relative_to(prefix)):
        raise RuntimeError(f"Expected a dedicated Meson package directory inside {prefix}: {site}")
    dist = metadata.PathDistribution(info)
    wheel = Parser().parsestr(dist.read_text("WHEEL") or "")
    requires_python = dist.metadata.get("Requires-Python")
    if (dist.metadata["Name"] != "meson" or wheel["Root-Is-Purelib"] != "true"
            or wheel.get_all("Tag") != ["py3-none-any"] or not requires_python
            or platform.python_version() not in SpecifierSet(requires_python)):
        raise RuntimeError(f"The bottled Meson package is not compatible with this Python: {info}")
    executable = shutil.which("meson")
    if not executable or Path(executable).resolve() != (prefix / "bin/meson").resolve(strict=True):
        raise RuntimeError(f"Meson is not the current Homebrew executable: {executable}")
    version = subprocess.check_output(
        [executable, "--version"], text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}).strip()
    if version != dist.version:
        raise RuntimeError(f"Meson executable/package version mismatch: {version} != {dist.version}")
    # A plain path admits the real package and metadata without executing .pth
    # hooks or importing packages from another interpreter's global site.
    bridge = purelib / "bottled-meson.pth"
    if bridge.is_symlink():
        raise RuntimeError(f"Expected a regular venv path file: {bridge}")
    contents = str(site) + "\n"
    if not bridge.exists() or bridge.read_text() != contents:
        bridge.write_text(contents)
    # Probe in a new interpreter so normal site initialization reads the .pth.
    # -B keeps this Python from writing bytecode into the Homebrew keg.
    probe = subprocess.check_output([
        sys.executable, "-I", "-B", "-c", """
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys
import mesonbuild
from mesonbuild import coredata

site, version, executable = sys.argv[1:]
dist = metadata.distribution("meson")
module = Path(mesonbuild.__file__).resolve()
core = Path(coredata.__file__).resolve()
if (dist.version != version or Path(dist.locate_file("")).resolve() != Path(site)
        or module.parent != Path(site, "mesonbuild")
        or core.parent != module.parent or coredata.version != version
        or Path(shutil.which("meson")).resolve() != Path(executable).resolve()):
    raise RuntimeError("Meson module, metadata or executable came from an unexpected location")
print(json.dumps({"version": dist.version, "site": site, "module": str(module),
                  "coredata": str(core), "module_version": coredata.version,
                  "executable": executable}))
""", str(site), version, executable], text=True)
    print(json.dumps(json.loads(probe), indent=2))


def data():
    """Check the published data bridge against the released Python packages."""
    root = Path(os.environ["SAGE_SHARE"]) / "sagemath"
    prefix = Path(subprocess.check_output(
        ["brew", "--prefix", "dimpase/tap/sagemath-deps"], text=True).strip()).resolve()
    directories = {
        "cremona": (files("elliptic_curve_databases") / "data" / "cremona", ("cremona_mini.db",)),
        "ellcurves": (files("elliptic_curve_databases") / "data" / "ellcurves", ("rank0",)),
        "graphs": (files("graph_databases") / "data", ("graphs.db", "isgci_sage.xml")),
        "reflexive_polytopes": (files("polytope_databases") / "data", ("Full3d/zzdb.info",)),
    }
    result = {}
    for name, (directory, payloads) in directories.items():
        path = Path(directory).resolve(strict=True)
        if not path.is_relative_to(prefix) or not (root / name).samefile(path):
            raise RuntimeError(f"{name} does not resolve to its bottled data: {path}")
        for filename in payloads:
            if not (path / filename).is_file():
                raise RuntimeError(f"Missing bottled data: {path / filename}")
        result[name] = str(path)
    print(json.dumps(result, indent=2))


def provenance():
    """Reject an installed Sage bottle accidentally taking precedence."""
    source = Path(os.environ["CI_SOURCE"]).resolve()
    build = Path(os.environ["CI_RUN_DIR"], "build").resolve()
    result = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.executable,
        "modules": {},
        "dependencies": {},
    }
    # Check both pure Python and compiled modules.
    for name, expected in (("sage.all", source), ("sage.rings.integer", build)):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(expected):
            raise RuntimeError(f"{name} came from {path}, expected it under {expected}")
        result["modules"][name] = str(path)
    dist = metadata.distribution("sagemath")
    direct_url = json.loads(dist.read_text("direct_url.json"))
    if direct_url != {"dir_info": {"editable": True}, "url": source.as_uri()}:
        raise RuntimeError(f"Not the candidate's editable installation: {direct_url}")
    for name, formula in (("sympy", "sagemath-deps"), ("sphinx", "sagemath-docbuild-deps")):
        prefix = Path(subprocess.check_output(
            ["brew", "--prefix", f"dimpase/tap/{formula}"], text=True).strip()).resolve()
        dist = metadata.distribution(name)
        path = Path(dist.locate_file("")).resolve()
        if not path.is_relative_to(prefix):
            raise RuntimeError(f"{name} came from {path}, expected the {formula} bottle")
        result["dependencies"][name] = {"version": dist.version, "path": str(path)}
    print(json.dumps(result, indent=2))


def doctest():
    paths = shlex.split(os.environ.get("CI_DOCTEST_PATHS", ""))
    if not paths:
        if os.environ.get("CI_SUITE", "smoke") == "all":
            paths = ["src/sage"]
        else:
            paths = [
                "src/sage/arith/misc.py",
                "src/sage/rings/integer.pyx",
                "src/sage/matrix/matrix_double_dense.pyx",
                "src/sage/libs/pari",
                "src/sage/libs/eclib",
                "src/sage/libs/gap",
                "src/sage/interfaces/maxima_lib.py",
            ]
    root = Path("src/sage").resolve()
    for value in paths:
        path = Path(value).resolve()
        if not path.is_relative_to(root) or not path.exists():
            raise ValueError(f"Expected an existing path under src/sage: {value}")
    arguments = ["-p", os.environ.get("CI_JOBS", "4"),
                 "--format", "github", "--random-seed=0", "--timeout=900",
                 f"--stats-path={os.environ['CI_RESULTS']}/doctest-stats.json", *paths]
    from sage.doctest.__main__ import main
    print("Running Sage doctests: " + shlex.join(arguments), flush=True)
    sys.argv = ["sage.doctest", *arguments]
    raise SystemExit(main())


if __name__ == "__main__":
    {"requirements": requirements, "meson": meson, "data": data,
     "provenance": provenance, "doctest": doctest}[sys.argv[1]]()
