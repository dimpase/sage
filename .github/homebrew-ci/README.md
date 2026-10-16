# Experimental CI with Homebrew dependency bottles

`homebrew-bottles.yml` builds the selected Sage checkout using the published
`dimpase/tap/sagemath-deps` and `dimpase/tap/sagemath-docbuild-deps` bottles.
The initial runner is `darwin-27` (Apple Silicon), using Homebrew Python 3.14
and four build/test workers.

The job creates a fresh venv with `--system-site-packages`, sources the bottles'
development environment, and installs pytest and pytest-xdist at the versions
in the candidate's `uv.lock`. It then builds editable sagelib with
`--no-build-isolation --no-deps` and disables Meson dependency fallbacks. The
candidate is not patched by the harness. Both pure Python and compiled Sage
imports are checked against the checkout and build directory.

For checkouts predating Sage PR #42878, `SAGE_DATA_PATH` points to temporary
symlinks to the bottled Cremona, elliptic-curve, graph, and polytope databases.
This uses Sage's existing data-directory configuration; it also works after
the data-package discovery changes. The exact paths are saved in the artifact.

The documentation dependencies provide Sphinx and its extensions, including
those needed by doctests. This job does not build the manuals or require TeX.
There is no separate `sagemath-doctest-deps` formula.

## Running the experiment

Pushes to `ci/homebrew-bottles-experiment` or `ci/homebrew-bottles/**` test that
branch. For example, make a branch from the experiment, apply a sagelib change,
and push it to your fork:

```sh
git switch -c ci/homebrew-bottles/my-change fork/ci/homebrew-bottles-experiment
git cherry-pick <commit-to-test>
git push fork HEAD
```

The workflow is registered in `dimpase/sage` and can also be dispatched from
the command line without merging it into the default branch:

```sh
gh workflow run homebrew-bottles.yml --repo dimpase/sage \
  --ref ci/homebrew-bottles-experiment \
  -f sage_repository=sagemath/sage -f sage_ref=develop \
  -f doctest_paths=src/sage/rings/integer.pyx
```

Use a branch, tag, commit, or `refs/pull/NUMBER/head` for `sage_ref`. For example,
`-f sage_ref=refs/pull/42920/head` tests that PR's checkout. Omit `doctest_paths`
to use the default smoke suite, or pass `-f suite=all` for the full suite.
Once the workflow is on the default branch, the Actions **Run workflow** form
will also expose these inputs.

Every run executes the pytest tests under `src/sage`. The default `smoke`
doctest suite covers arithmetic, integer and numerical matrices, PARI, eclib,
GAP, and Maxima. Select `all` for all doctests under `src/sage`, or supply
space-separated paths such as:

```text
src/sage/symbolic/rootsum.py src/sage/interfaces/sympy.py
```

Use `all` or relevant paths to assess changes outside the smoke suite. Failures
are not suppressed; the doctests still run after a pytest failure. Optional
tests use Sage's normal feature detection. Missing optional software can be
reported as skipped tests.

On macOS the launcher imports psutil before starting Sage's doctest workers.
The bottled psutil 7.2.2 initializes IOKit during import; Sage's doctest timer
otherwise first imports it in a forked worker when timing a pexpect interface.
That caused Objective-C initialization aborts in the GAP doctests on darwin-27.
Preloading avoids that late initialization without disabling macOS fork checks.

The initial smoke run on Sage 11.0.beta2 found PARI-related doctest failures
with the installed dependencies. They remain visible failures while the
corresponding Sage fixes are pending; the harness does not alter expectations
or suppress their exit status.

The artifact contains build and test logs, pytest JUnit results, doctest
statistics, Meson diagnostics, tap commits, package versions, and import
provenance. The venv and build directory are removed afterwards; the dependency
bottles remain installed for subsequent runs.

The runner must have Homebrew and Xcode command-line tools. The setup updates
Homebrew metadata and unlinks core NumPy, SciPy, or python-matplotlib when
necessary to install the tap's declared conflicting variants. Core m4ri and
m4rie remain the selected dependencies. Runs of this workflow are serialized.
Other jobs sharing the machine must not change its Homebrew installation while
this job runs.

This experiment deliberately has no automatic public pull-request trigger:
a maintainer selects code to run on the persistent self-hosted runner. Linux
can be added as a separate runner after the macOS experiment is validated.
