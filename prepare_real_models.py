"""Check pinned input files; download only when explicitly passed --download.

Uses the published experiment manifests, not a floating main branch.
No model loading, training, credentials or paid service is required here.
"""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['all', 'retrieval', 'finetune', 'cache'], default='all')
    parser.add_argument('--cache-root', type=Path, default=Path.home() / '.cache/huggingface/hub')
    parser.add_argument('--download', action='store_true', help='Explicitly allow downloading the recorded input files')
    args = parser.parse_args()
    failures = []
    kinds = ['retrieval', 'finetune', 'cache'] if args.experiment == 'all' else [args.experiment]
    for kind in kinds:
        manifest = json.loads((BASE / 'real-experiments/results' / (kind + '.json')).read_text(encoding='utf8'))['model']
        repo, revision = manifest['repo'], manifest['revision']
        root = args.cache_root / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
        for filename, expected in manifest['files'].items():
            filename = filename.replace('\\', '/')
            target = root / filename
            if args.download and not target.is_file():
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=repo, filename=filename, revision=revision, cache_dir=str(args.cache_root))
            if not target.is_file():
                failures.append(f'MISSING {repo}@{revision}: {filename}')
            elif target.stat().st_size != expected['bytes'] or digest(target) != expected['sha256']:
                failures.append(f'HASH MISMATCH {repo}@{revision}: {filename}; preserve this file for diagnosis')
        print(f'Checked {kind}: {repo}@{revision}')
    if failures:
        print('\n'.join(failures))
        raise SystemExit(1)
    print('All recorded model inputs match the published SHA-256 manifests.')


if __name__ == '__main__':
    main()
