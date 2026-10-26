"""Install pinned Isaac Lab skills with their repository-relative references."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess

REVISION = 'a9fb50e5388724a680bbc5c1f836917271605a99'
REPOSITORY = 'https://github.com/isaac-sim/IsaacLab.git'
SUBTREES = ['skills', 'docs', 'source', 'scripts', 'tools', '.github', '.agents', '.claude']


def git(*args):
    return subprocess.check_output(['git', *map(str, args)], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=Path, help='Reuse a prepared pinned checkout.')
    args = parser.parse_args()
    codex_root = Path(os.environ.get('CODEX_HOME', Path.home() / '.codex'))
    source = codex_root / 'skill-sources' / ('isaaclab-' + REVISION[:12])
    destination = Path.home() / '.agents' / 'skills'
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        if args.seed:
            if git('-C', args.seed, 'rev-parse', 'HEAD') != REVISION:
                raise RuntimeError('Seed does not match the pinned revision.')
            shutil.copytree(args.seed, source, symlinks=True)
        else:
            git('init', source)
            git('-C', source, 'remote', 'add', 'origin', REPOSITORY)
            git('-C', source, 'sparse-checkout', 'init', '--cone')
    if git('-C', source, 'remote', 'get-url', 'origin') != REPOSITORY:
        raise RuntimeError('Unexpected source remote; refusing to modify it.')
    try:
        current = git('-C', source, 'rev-parse', 'HEAD')
    except subprocess.CalledProcessError:
        current = None
    if current is None:
        git('-C', source, 'fetch', '--depth', '1', '--filter=blob:none', 'origin', REVISION)
        git('-C', source, 'checkout', '--detach', REVISION)
    elif current != REVISION:
        raise RuntimeError('Existing source revision differs; refusing to overwrite.')
    git('-C', source, 'sparse-checkout', 'set', *SUBTREES)
    entries = []
    for skill_file in sorted((source / 'skills').glob('*/*/SKILL.md')):
        names = [line[6:].strip() for line in skill_file.read_text().splitlines() if line.startswith('name: ')]
        if len(names) != 1 or not names[0].startswith('isaaclab-') or '/' in names[0]:
            raise RuntimeError(f'Invalid skill name: {skill_file}')
        target = skill_file.parent
        link = destination / names[0]
        if link.is_symlink() or link.exists():
            if not link.is_symlink() or link.resolve() != target.resolve():
                raise RuntimeError(f'Existing skill conflicts: {link}')
        entries.append((link, target))
    if len(entries) != 24 or len({link for link, _ in entries}) != 24:
        raise RuntimeError('Expected exactly 24 distinct skills in the pinned checkout.')
    destination.mkdir(parents=True, exist_ok=True)
    for link, target in entries:
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(target, destination), target_is_directory=True)
        if not (link / 'SKILL.md').is_file():
            raise RuntimeError(f'Broken installed skill: {link}')
        print(link.name)
    print(f'Verified {len(entries)} skills at {destination}; source revision {REVISION}')


if __name__ == '__main__':
    main()
