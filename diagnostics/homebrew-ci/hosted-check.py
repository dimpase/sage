"""Evidence checks for one disposable hosted macOS helper migration run."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys


ROOTS = {'dimpase/tap/sagemath-deps': '11.0.beta6_2',
         'dimpase/tap/sagemath-docbuild-deps': '11.0.beta6_1'}
BASE = '026e931e47064f0a7fc095580119365518bfe7c0'
PREVIOUS_FIX = 'c96258a1038f7386128969e1f3a5bdf7b0b09eab'
FIX = '173627b8c3376b20a4f4a56978ee2a7a811a98cb'
HELPERS = {'run.sh': '5dc84fb99b6ad86c30724b25eb56b215c8a5a5a464dbfd36be9c670defe6988f',
           'check.py': '3ef4d1cc5e4cd95c1d6ca9166fcc33e5447c03850f40618e0fe583cf998df3ab'}
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


def meson_keg_state():
    row = receipt('meson')
    check_bottle(row, 'homebrew/core')
    keg = Path(row['keg'])
    entries = {}
    for path in sorted(keg.rglob('*')):
        details = path.lstat()
        entry = {'mode': stat.S_IMODE(details.st_mode), 'mtime_ns': details.st_mtime_ns}
        if path.is_symlink():
            entry.update(kind='symlink', target=os.readlink(path))
        elif path.is_file():
            entry.update(kind='file', size=details.st_size, sha256=digest(path))
        elif path.is_dir():
            entry.update(kind='directory')
        else:
            raise RuntimeError(f'Unexpected Meson keg entry: {path}')
        entries[str(path.relative_to(keg))] = entry
    return {'receipt': row, 'entries': entries}


def meson_bridge_state(provenance_name):
    out = Path(os.environ['CI_RESULTS'])
    record = json.loads((out / provenance_name).read_text())
    require(set(record) == {'version', 'site', 'module', 'coredata', 'module_version', 'executable'},
            'Unexpected Meson provenance')
    row = receipt('meson')
    keg = Path(row['keg'])
    site = Path(record['site']).resolve(strict=True)
    require(site.is_relative_to(keg), 'Meson package is outside the current bottle')
    require(record['version'] == row['receipt']['source']['versions']['stable'],
            'Meson metadata differs from the installed formula version')
    require(Path(record['module']).resolve(strict=True) == site / 'mesonbuild/__init__.py',
            'Unexpected imported Meson module')
    require(Path(record['coredata']).resolve(strict=True) == site / 'mesonbuild/coredata.py'
            and record['module_version'] == record['version'], 'Unexpected Meson coredata/version')
    require(Path(record['executable']).resolve(strict=True) == keg / 'bin/meson',
            'Meson executable changed')
    venv = Path(os.environ['CI_RUN_DIR'], 'venv').resolve(strict=True)
    paths = list(venv.glob('lib/python*/site-packages/bottled-meson.pth'))
    require(len(paths) == 1, f'Expected one venv Meson bridge: {paths}')
    path = paths[0]
    require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(venv),
            'Meson bridge escaped the venv')
    require(path.read_text() == str(site) + '\n', 'The Meson bridge must be one plain path')
    details = path.stat()
    return {'provenance': record, 'bridge_path': str(path), 'bridge_sha256': digest(path),
            'bridge_mtime_ns': details.st_mtime_ns, 'bridge_mode': stat.S_IMODE(details.st_mode)}


def check_meson_preserved(stage, provenance_name):
    out = Path(os.environ['CI_RESULTS'])
    before = json.loads((out / 'meson-keg-before-venv.json').read_text())
    after = meson_keg_state()
    save(f'meson-keg-{stage}.json', after)
    require(after == before, 'Meson bottle contents/bytecode/metadata changed')
    bridge = meson_bridge_state(provenance_name)
    save(f'meson-bridge-{stage}.json', bridge)
    if stage != 'before-repeat':
        previous = json.loads((out / 'meson-bridge-before-repeat.json').read_text())
        require(bridge == previous, 'Meson bridge or provenance changed on repeat/final check')


def source_state():
    source = Path(os.environ['CI_SOURCE'])
    helpers = Path(os.environ['CI_HELPERS'])
    head = command('git', 'rev-parse', 'HEAD', cwd=source)
    require(head == os.environ['GITHUB_SHA'], 'Source differs from the diagnostic workflow commit')
    require(command('git', 'rev-parse', 'HEAD^', cwd=source) == FIX,
            'Diagnostic must be a direct normal descendant of the reviewed fix')
    require(command('git', 'rev-parse', f'{FIX}^', cwd=source) == PREVIOUS_FIX,
            'Reviewed metadata fix must be a normal child of the migration fix')
    require(command('git', 'rev-parse', f'{PREVIOUS_FIX}^', cwd=source) == BASE,
            'The migration fix must retain the unchanged beta6 base')
    fix_changes = set(command('git', 'diff', '--name-only', BASE, FIX, cwd=source).splitlines())
    require(fix_changes == {'.github/homebrew-ci/run.sh', '.github/homebrew-ci/check.py'},
            f'Unexpected product changes relative to beta6: {fix_changes}')
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
    return {'head': head, 'reviewed_fix': FIX, 'previous_fix': PREVIOUS_FIX,
            'unchanged_sage_base': BASE,
            'changed_paths_from_base': sorted(changed), 'helper_sha256': hashes,
            'source_status': command('git', 'status', '--short', cwd=source)}


def main(phase):
    require(os.environ.get('RUNNER_ENVIRONMENT') == 'github-hosted', 'Hosted runner required')
    require(platform.system() == 'Darwin' and platform.machine() == 'arm64', 'macOS ARM64 required')
    require(os.environ['GITHUB_REPOSITORY'] == 'dimpase/sage', 'Unexpected repository')
    require(os.environ['GITHUB_REF'] == 'refs/heads/diagnostics/homebrew-ci-compat-v2', 'Wrong diagnostic branch')
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
            check_meson_preserved('final', 'meson-package-repeat.json')
            before = json.loads((out / 'source-before.json').read_text())
            after = source_state()
            save('source-after.json', after)
            require(before == after, 'Candidate source/helper changed during validation')
    elif phase == 'meson-before-venv':
        save('meson-keg-before-venv.json', meson_keg_state())
    elif phase == 'meson-before-repeat':
        check_meson_preserved('before-repeat', 'meson-package.json')
    elif phase == 'meson-after-repeat':
        check_meson_preserved('after-repeat', 'meson-package-repeat.json')
    else:
        raise ValueError(phase)
    save(f'{phase}-check.json', {'status': 'passed', 'phase': phase})


if __name__ == '__main__':
    main(sys.argv[1])
