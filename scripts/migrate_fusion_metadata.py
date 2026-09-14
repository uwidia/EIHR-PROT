"""Migrate fusion metadata and dependent checksums without changing model tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cohort_hash(ids):
    return hashlib.sha256('\0'.join(sorted(ids)).encode()).hexdigest()


def transform(value, hashes):
    if isinstance(value, dict):
        result, changed = value.copy(), False
        for key in ('excluded_query_ids', 'validation_exclude_ids'):
            if key in result:
                result['validation_policy_sha256'] = cohort_hash(result.pop(key))
                changed = True
        for key in ('removed_original_rows', 'removed_enriched_rows', 'original_query_count', 'validation_excluded'):
            if key in result:
                del result[key]
                changed = True
        if 'remaining_query_count' in result:
            result['query_count'] = result.pop('remaining_query_count')
            changed = True
        if result.get('fusion_protocol_version') == 'validation_exclude_two_v1':
            result['fusion_protocol_version'] = 'fusion_v1'
            changed = True
        for key, item in list(result.items()):
            result[key], different = transform(item, hashes)
            changed |= different
        return result, changed
    if isinstance(value, (list, tuple)):
        pairs = [transform(item, hashes) for item in value]
        return type(value)(item for item, _ in pairs), any(flag for _, flag in pairs)
    if isinstance(value, str):
        updated = hashes.get(value, value)
        for version in (1, 2):
            updated = updated.replace(f'validation_exclude_two_v{version}', f'cohort_v{version}')
        return updated, updated != value
    return value, False


def tensor_payloads(path):
    with zipfile.ZipFile(path) as archive:
        return {name.split('/', 1)[1]: hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist() if '/data/' in name}


def recover_metadata_links(value, root, hashes):
    if isinstance(value, dict):
        pairs = []
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            pairs.append((value['path'], value['sha256']))
        for name, key in (('evidence_report', 'evidence_report_sha256'), ('audit_path', 'audit_sha256')):
            if isinstance(value.get(name), str) and isinstance(value.get(key), str):
                pairs.append((value[name], value[key]))
        for location, old_hash in pairs:
            location = location.replace('\\', '/')
            if '/runs/' in location:
                location = location.split('/runs/', 1)[1]
            elif location.startswith('runs/'):
                location = location[5:]
            else:
                continue
            for version in (1, 2):
                location = location.replace(f'cohort_v{version}', f'validation_exclude_two_v{version}')
            target = (root / location).resolve()
            if target.is_relative_to(root) and target.suffix == '.json' and target.is_file():
                if transform(json.loads(target.read_text(encoding='utf-8')), {})[1]:
                    continue
                actual = digest(target)
                if actual != old_hash:
                    hashes[old_hash] = actual
        for item in value.values():
            recover_metadata_links(item, root, hashes)
    elif isinstance(value, (list, tuple)):
        for item in value:
            recover_metadata_links(item, root, hashes)


def migrate(root, *, recover_links=False):
    import torch
    root = root.resolve(strict=True)
    journal = root / '.fusion_metadata_migration.json'
    files = sorted(p for p in root.rglob('*.json') if p != journal)
    files += sorted(p for p in root.rglob('*.pt') if any(
        part in ('sequence_homology_fixed_fusion', 'sequence_homology_identity_fusion') for part in p.parts))
    for directory in root.rglob('*'):
        if directory.is_dir() and directory.name in ('validation_exclude_two_v1', 'validation_exclude_two_v2'):
            destination = directory.with_name(directory.name.replace('validation_exclude_two_', 'cohort_'))
            if destination.exists():
                raise FileExistsError(destination)
    original_hashes = {p: digest(p) for p in files}
    replacements = json.loads(journal.read_text()) if journal.exists() else {}
    staged = {}
    if recover_links:
        for path in files:
            value = (json.loads(path.read_text(encoding='utf-8')) if path.suffix == '.json'
                     else torch.load(path, map_location='cpu', weights_only=False))
            recover_metadata_links(value, root, replacements)
        journal.write_text(json.dumps(replacements, sort_keys=True))
    with tempfile.TemporaryDirectory(prefix='fusion_metadata_') as temp:
        stage = Path(temp)
        for iteration in range(12):
            previous = dict(replacements)
            for index, path in enumerate(files):
                if path.suffix == '.json':
                    value = json.loads(path.read_text(encoding='utf-8'))
                else:
                    value = torch.load(path, map_location='cpu', weights_only=False)
                updated, changed = transform(value, replacements)
                if not changed:
                    continue
                if isinstance(updated, dict) and updated.get('fusion_protocol_version') == 'cohort_v1':
                    updated['fusion_protocol_version'] = 'fusion_v1'
                target = stage / str(index) / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == '.json':
                    target.write_text(json.dumps(updated, indent=2, sort_keys=True), encoding='utf-8')
                else:
                    torch.save(updated, target)
                    if tensor_payloads(path) != tensor_payloads(target):
                        raise ValueError('Checkpoint tensor payload changed during metadata migration')
                replacements[original_hashes[path]] = digest(target)
                staged[path] = target
            print(f'Metadata pass {iteration + 1}: {len(staged)} artifacts staged', flush=True)
            if previous == replacements:
                break
        else:
            raise ValueError('Metadata checksum dependencies did not converge')
        for path, target in staged.items():
            if digest(path) != original_hashes[path]:
                raise ValueError('An artifact changed while migration was being prepared')
        journal.write_text(json.dumps(replacements, sort_keys=True))
        for path, target in staged.items():
            # Commit from a sibling file so replacement stays on the same volume.
            sibling = path.with_name(path.name + '.metadata-tmp')
            with target.open('rb') as source, sibling.open('wb') as dest:
                import shutil
                shutil.copyfileobj(source, dest)
            os.replace(sibling, path)
        directories = sorted((p for p in root.rglob('*') if p.is_dir()
                              and p.name in ('validation_exclude_two_v1', 'validation_exclude_two_v2')),
                             key=lambda p: len(p.parts), reverse=True)
        for directory in directories:
            destination = directory.with_name(directory.name.replace('validation_exclude_two_', 'cohort_'))
            if not directory.resolve().is_relative_to(root) or not destination.resolve().is_relative_to(root):
                raise ValueError('Directory migration escaped the selected root')
            if destination.exists():
                raise FileExistsError(destination)
            directory.rename(destination)
    journal.unlink(missing_ok=True)
    print(f'Updated {len(staged)} metadata artifacts; checkpoint tensor payloads verified unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs'))
    parser.add_argument('--recover-links', action='store_true', help='Repair metadata links after an interrupted migration without a journal.')
    args = parser.parse_args()
    migrate(args.root, recover_links=args.recover_links)
