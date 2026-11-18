"""Trusted offline compile entry point, executed only inside the pinned image."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main(source):
    source = Path(source).resolve()
    tree = Path('/opt/opencode-source')
    if not (source / 'packages/opencode/script/build.ts').is_file():
        raise RuntimeError('Complete candidate source is required')
    # Remove original release files so deletions in a candidate are respected.
    # Retain only the image's installed dependency trees, including their original
    # relative workspace links. Those links now resolve to the candidate source.
    for path in sorted(tree.rglob('*'), reverse=True):
        if 'node_modules' in path.relative_to(tree).parts:
            continue
        if path.is_file() or path.is_symlink():
            path.unlink()
    shutil.copytree(source, tree, dirs_exist_ok=True)
    subprocess.run(['bun', 'run', 'packages/opencode/script/build.ts', '--single',
                    '--skip-install', '--skip-embed-web-ui'], cwd=tree, check=True)
    binary = tree / 'packages/opencode/dist/opencode-linux-x64/bin/opencode'
    if not binary.is_file() or binary.is_symlink():
        raise RuntimeError('Compiler did not produce a regular executable')
    subprocess.run([str(binary), '--version'], cwd='/tmp', check=True)
    print(json.dumps({'compiled': True, 'source': str(source), 'binary': str(binary)}))


if __name__ == '__main__':
    main(sys.argv[1])
