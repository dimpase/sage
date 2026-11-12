"""Evidence checks for one disposable hosted macOS helper migration run."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


ROOTS = {'dimpase/tap/sagemath-deps': '11.0.beta6_2',
         'dimpase/tap/sagemath-docbuild-deps': '11.0.beta6_1'}
BASE = '026e931e47064f0a7fc095580119365518bfe7c0'
FIX = 'c96258a1038f7386128969e1f3a5bdf7b0b09eab'
HELPERS = {'run.sh': 'c4ed3a99946a5903629f65f445bd2d95a2ff26abd95b1d6a2b4417954329a233',
           'check.py': 'fa6f382aa62454871384dd95ad9561925a2e34661f7528409ba503f7a4bdbbec'}
MIGRATION = re.compile(r'^\++ brew reinstall --force-bottle dimpase/tap/gmp-ecm$', re.M)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (Path(os.environ['CI_RESULTS']) / name).write_text(json.dumps(value, indent=2) + '\n')


def receipt(name):
    prefix = Path(command('brew', '--prefix')).resolve(strict=True)
    short = name.rsplit('/', 1)[-1]
    opt = prefix / 'opt' / short
    require(opt.is_symlink(), f'Missing opt link: {opt}')
    keg = opt.resolve(strict=True)
    require(keg.parent == prefix / 'Cellar' / short, f'Unexpected opt target: {opt} -> {keg}')
    path = keg / 'INSTALL_RECEIPT.json'
    data = json.loads(path.read_text())
    return {'name': name, 'opt': str(opt), 'keg': str(keg), 'pkg_version': keg.name,
            'receipt_sha256': digest(path), 'receipt': data}


def check_bottle(row, owner):
    require(row['receipt']['source']['tap'] == owner,
            f'Wrong owner for {row["name"]}: {row["receipt"]["source"]}')
    require(row['receipt']['poured_from_bottle'] is True,
            f'Not poured from a bottle: {row["name"]}')


def gmp_snapshot(stage, owner):
    row = receipt('dimpase/tap/gmp-ecm' if owner == 'dimpase/tap' else 'gmp-ecm')
    prefix = Path(command('brew', '--prefix')).resolve(strict=True)
    linked = prefix / 'var/homebrew/linked/gmp-ecm'
    binary = prefix / 'bin/ecm'
    row.update(linked=linked.is_symlink(),
               linked_keg=str(linked.resolve()) if linked.is_symlink() else None,
               ecm_binary=str(binary.resolve()) if binary.exists() else None)
    save(f'gmp-{stage}.json', row)
    check_bottle(row, owner)
    if owner == 'dimpase/tap':
        require(row['pkg_version'] == '7.0.7_1', 'Expected the published tap gmp-ecm revision1')
        require(row['linked_keg'] == row['keg'] and row['ecm_binary'] == str(Path(row['keg']) / 'bin/ecm'),
                'The migrated tap keg is not linked into the prefix')
    elif stage == 'core-linked':
        require(row['linked_keg'] == row['keg'], 'The initial core install was not linked')
    elif stage == 'core':
        require(not row['linked'] and row['ecm_binary'] is None,
                'The diagnostic must reproduce the unlinked core-keg state')
    return row


def closure(stage):
    pending = list(ROOTS)
    seen = {}
    try:
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            row = receipt(name)
            seen[name] = row
            owner = '/'.join(name.split('/')[:-1]) if '/' in name else 'homebrew/core'
            check_bottle(row, owner)
            if name in ROOTS:
                require(row['pkg_version'] == ROOTS[name], f'Wrong root revision: {name}')
            pending.extend(dep['full_name'] for dep in row['receipt']['runtime_dependencies'])
        require('dimpase/tap/gmp-ecm' in seen, 'Tap gmp-ecm absent from the dependency closure')
        require('gmp-ecm' not in seen, 'Core gmp-ecm still recorded in the dependency closure')
    finally:
        save(f'closure-{stage}.json', seen)
    return seen


def preserve_dependency_records(stage):
    out = Path(os.environ['CI_RESULTS'])
    for name in ('brew-versions.txt', 'dependency-bottles.json', 'tap-commits.txt'):
        shutil.copy2(out / name, out / f'{stage}-{name}')


def source_state():
    source = Path(os.environ['CI_SOURCE'])
    helpers = Path(os.environ['CI_HELPERS'])
    head = command('git', 'rev-parse', 'HEAD', cwd=source)
    require(head == os.environ['GITHUB_SHA'], 'Source differs from the diagnostic workflow commit')
    require(command('git', 'rev-parse', 'HEAD^', cwd=source) == FIX,
            'Diagnostic must be a direct normal descendant of the reviewed fix')
    require(command('git', 'rev-parse', f'{FIX}^', cwd=source) == BASE,
            'Reviewed fix must be based directly on unchanged beta6')
    changed = set(command('git', 'diff', '--name-only', BASE, 'HEAD', cwd=source).splitlines())
    allowed = {'.github/homebrew-ci/run.sh', '.github/homebrew-ci/check.py',
               '.github/workflows/homebrew-ci-compat-diagnostic.yml',
               'diagnostics/homebrew-ci/hosted-check.py'}
    require(changed == allowed, f'Unexpected source changes relative to beta6: {changed}')
    require(not command('git', 'diff', '--name-only', 'HEAD', cwd=source), 'Tracked source changed')
    hashes = {}
    for name in ('run.sh', 'check.py'):
        path = helpers / name
        hashes[name] = digest(path)
        require(hashes[name] == HELPERS[name], f'Unexpected reviewed helper bytes: {name}')
        require(hashes[name] == digest(source / '.github/homebrew-ci' / name),
                f'Helper differs between exact-commit checkouts: {name}')
    return {'head': head, 'reviewed_fix': FIX, 'unchanged_sage_base': BASE,
            'changed_paths_from_base': sorted(changed), 'helper_sha256': hashes,
            'source_status': command('git', 'status', '--short', cwd=source)}


def main(phase):
    require(os.environ.get('RUNNER_ENVIRONMENT') == 'github-hosted', 'Hosted runner required')
    require(platform.system() == 'Darwin' and platform.machine() == 'arm64', 'macOS ARM64 required')
    require(os.environ['GITHUB_REPOSITORY'] == 'dimpase/sage', 'Unexpected repository')
    require(os.environ['GITHUB_REF'] == 'refs/heads/diagnostics/homebrew-ci-compat', 'Wrong diagnostic branch')
    out = Path(os.environ['CI_RESULTS'])
    if phase == 'prepare':
        save('source-before.json', source_state())
        save('runner.json', {name: os.environ.get(name) for name in (
            'RUNNER_ENVIRONMENT', 'RUNNER_NAME', 'RUNNER_OS', 'RUNNER_ARCH',
            'ImageOS', 'ImageVersion', 'GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT')})
        for name in ('run.sh', 'check.py'):
            shutil.copy2(Path(os.environ['CI_HELPERS']) / name, out / ('helper-' + name))
        shutil.copy2(__file__, out / 'diagnostic-hosted-check.py')
        (out / 'brew-config.txt').write_text(command('brew', 'config') + '\n')
        (out / 'initial-installed-info.json').write_text(command('brew', 'info', '--json=v2', '--installed') + '\n')
    elif phase in ('core-linked', 'core'):
        gmp_snapshot(phase, 'homebrew/core')
    elif phase == 'migrated':
        before = json.loads((out / 'gmp-core.json').read_text())
        check_bottle(before, 'homebrew/core')
        gmp_snapshot('migrated', 'dimpase/tap')
        require(len(MIGRATION.findall((out / 'dependencies-first.log').read_text())) == 1,
                'Expected exactly one explicit core-to-tap reinstall')
        preserve_dependency_records(phase)
        closure('migrated')
    elif phase in ('idempotent', 'final'):
        before = json.loads((out / 'gmp-migrated.json').read_text())
        after = gmp_snapshot(phase, 'dimpase/tap')
        require(before == after, 'The already-tap gmp-ecm keg/receipt changed')
        require(not MIGRATION.search((out / 'dependencies-second.log').read_text()),
                'The already-tap dependency phase attempted another migration')
        if phase == 'idempotent':
            preserve_dependency_records(phase)
        closure(phase)
        if phase == 'final':
            before = json.loads((out / 'source-before.json').read_text())
            after = source_state()
            save('source-after.json', after)
            require(before == after, 'Candidate source/helper changed during validation')
    else:
        raise ValueError(phase)
    save(f'{phase}-check.json', {'status': 'passed', 'phase': phase})


if __name__ == '__main__':
    main(sys.argv[1])
