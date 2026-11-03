"""Delete explicitly superseded archive payloads, preserving provenance metadata."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PROJECT = Path('/Users/qiqing/learn_resource/mutil-language/pitfalls-cross-lingual')
WORKTREE = Path('/Users/qiqing/.codex/worktrees/762b/pitfalls-cross-lingual')
ARCHIVE = WORKTREE / 'analysis_output/_obsolete_archive/relation_review_replaced_20261023'
AUDIT = Path(__file__).resolve().parent
NAMES = {'relation_expansion_20261022', 'relation_expansion_full_20261022_v1',
         'relation_expansion_recovery_20261023_v2', 'relation_expansion_recovery_20261023_v3',
         'relation_expansion_repair_20261023'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def now():
    return datetime.now(timezone.utc).isoformat()


def targets(manifest):
    result = []
    for entry in manifest['entries']:
        target = Path(entry['recoverable_archive_path'])
        assert target.name in NAMES and target.parent.name in {'primary', 'worktree'}
        assert target.parent.parent == ARCHIVE and target.resolve() == target
        assert not target.is_symlink()
        result.append(target)
    assert len(result) == len(set(result)) == 10
    return result


def verify_payloads(manifest):
    count = size = 0
    for entry, target in zip(manifest['entries'], targets(manifest)):
        assert target.is_dir(), str(target)
        expected = {v['path']: v for v in entry['file_bindings']}
        actual = {str(p.relative_to(target)): p for p in target.rglob('*') if p.is_file() or p.is_symlink()}
        assert actual.keys() == expected.keys(), 'Archive inventory changed: ' + str(target)
        for name, path in actual.items():
            assert not path.is_symlink(), 'Unexpected symlink: ' + str(path)
            binding = expected[name]
            assert path.stat().st_size == binding['bytes'] and sha(path) == binding['sha256'], str(path)
            count += 1
            size += binding['bytes']
    return count, size


def active_bindings():
    pointer_path = PROJECT / 'analysis_output/active_data_pipeline.json'
    pointer = read(pointer_path)
    run = Path(pointer['run_dir'])
    manifest_path = run / 'manifest.json'
    assert sha(manifest_path) == pointer['manifest_sha256']
    manifest = read(manifest_path)
    bindings = {str(pointer_path): sha(pointer_path), str(manifest_path): sha(manifest_path)}
    bindings.update(manifest['bindings'])
    for name, expected in bindings.items():
        assert not Path(name).is_relative_to(ARCHIVE)
        assert sha(Path(name)) == expected, name
    for root in (PROJECT, WORKTREE):
        for relative in ['utils/.env',
                         'data_processed/factual_triples/triple-full-v1/models/qwen3.7-plus/triple_extractions.jsonl',
                         'data_processed/factual_triples/triple-full-v1/models/sensenova-6.7-flash-lite/triple_extractions.jsonl',
                         'data_processed/factual_triples/public-benchmarks-full-v1/models/qwen3.7-plus/triple_extractions.jsonl']:
            path = root / relative
            if path.exists():
                bindings[str(path)] = sha(path)
    # Actual bound inputs and runtime must contain no dependency on archive payloads.
    dependencies = []
    for name in manifest['bindings']:
        path = Path(name)
        content = path.read_text()
        if str(ARCHIVE) in content or '_obsolete_archive/' in content:
            dependencies.append(name)
    assert not dependencies, 'Current bound artifacts reference archived payloads'
    return bindings, run


def retained_exposure(manifest):
    """Keep contact identifiers/hashes, never proposals, verdicts, reasons or QA text.

    Observed outputs certify recorded contact only; failed requests do not prove
    actual model exposure. Missing failed-request IDs remain explicitly unknown.
    """
    entries = manifest['entries']
    contacts = {}
    requests = []
    unresolved_dispatches = 0

    def add(row, model, role, evidence_hash, agent=None):
        identifier = row.get('record_id')
        if not identifier or not model:
            return
        source_hash = row.get('input_record_sha256') or row.get('original_record_sha256')
        key = identifier, model, role, source_hash
        item = contacts.setdefault(key, {'record_id': identifier, 'requested_model': model,
                                        'stage': role, 'input_record_sha256': source_hash,
                                        'evidence_artifact_sha256s': set(), 'recorded_agent_task_ids': set()})
        item['evidence_artifact_sha256s'].add(evidence_hash)
        if agent:
            item['recorded_agent_task_ids'].add(agent)

    def relocate(value):
        candidate = Path(value)
        for entry in entries:
            original = Path(entry['removed_active_path'])
            if candidate.is_relative_to(original):
                return Path(entry['recoverable_archive_path']) / candidate.relative_to(original)
        if candidate.is_relative_to(ARCHIVE):
            return candidate
        return None

    visited_outputs = set()
    for entry in entries:
        root = Path(entry['recoverable_archive_path'])
        for binding in entry['file_bindings']:
            path = root / binding['path']
            if path.suffix == '.json' and path.name.startswith('batch_'):
                value = read(path)
                for row in value.get('records', []):
                    if isinstance(row, dict):
                        add(row, row.get('requested_model'), 'qwen_relation_proposal_output', binding['sha256'])
                attempts = value.get('attempts', [])
                if attempts:
                    requests.append({'artifact_sha256': binding['sha256'],
                                     'batch_input_sha256': value.get('input_sha256'),
                                     'base_prompt_sha256': value.get('prompt_sha256'),
                                     'batch_index': value.get('batch_index'),
                                     'attempt_count': len(attempts),
                                     'attempt_status_counts': dict(Counter(a.get('status', 'unknown') for a in attempts)),
                                     'source_ids_for_failed_attempts_not_reconstructed': True})
            if path.name == 'proposals.jsonl':
                for row in rows(path):
                    add(row, row.get('requested_model'), 'qwen_relation_proposal_output', binding['sha256'])
            if path.suffix == '.json' and path.name.startswith('dispatch'):
                value = read(path)
                tasks = value.get('dispatches', []) if isinstance(value, dict) else []
                for task in tasks:
                    if not isinstance(task, dict) or not task.get('output_path'):
                        continue
                    output = relocate(task['output_path'])
                    if output is None or not output.exists():
                        unresolved_dispatches += 1
                        continue
                    if not output.is_relative_to(ARCHIVE):
                        raise ValueError('Exposure output outside archive')
                    model, role, agent = task.get('requested_model'), task.get('role'), task.get('agent_id')
                    identity = str(output), model, role, agent
                    if identity in visited_outputs:
                        continue
                    visited_outputs.add(identity)
                    output_hash = sha(output)
                    for row in rows(output):
                        add(row, model, 'luna_relation_review' if role == 'reviewer' else 'sol_relation_adjudication', output_hash, agent)
    values = []
    for item in contacts.values():
        values.append({**item, 'evidence_artifact_sha256s': sorted(item['evidence_artifact_sha256s']),
                       'recorded_agent_task_ids': sorted(item['recorded_agent_task_ids']),
                       'hidden_model_snapshot_verified': False, 'human_gold': False})
    values.sort(key=lambda v: (v['record_id'], v['requested_model'], v['stage'], v['input_record_sha256'] or ''))
    contact_path = AUDIT / 'historical_model_contacts.jsonl'
    contact_path.write_text(''.join(json.dumps(v, ensure_ascii=False, sort_keys=True) + '\n' for v in values))
    request_path = AUDIT / 'historical_request_metadata.jsonl'
    # Archive copies may repeat requests; retain unique metadata but do not claim call totals.
    unique_requests = {json.dumps(v, sort_keys=True): v for v in requests}
    request_path.write_text(''.join(json.dumps(v, sort_keys=True) + '\n' for v in unique_requests.values()))
    write(AUDIT / 'historical_exposure_summary.json', {
        'created_at': now(), 'source_archive_manifest_sha256': sha(ARCHIVE / 'archive_manifest.json'),
        'unique_source_ids_with_recorded_contact': len({v['record_id'] for v in values}),
        'contact_metadata_rows': len(values), 'unresolved_dispatch_output_references': unresolved_dispatches,
        'known_contact_stages': dict(Counter(v['stage'] for v in values)),
        'exposure_closure_complete': False, 'failed_or_missing_source_exposure_remains_unknown': True,
        'deleting_outputs_does_not_make_sources_clean_holdout': True,
        'semantic_proposals_reviews_verdicts_reasons_or_QA_preserved_here': False,
        'contact_inventory_sha256': sha(contact_path), 'request_metadata_sha256': sha(request_path),
        'human_gold': False})


def prepare():
    manifest = read(ARCHIVE / 'archive_manifest.json')
    count, size = verify_payloads(manifest)
    guards, run = active_bindings()
    retained_exposure(manifest)
    write(AUDIT / 'protected_guards.json', guards)
    plan = {'prepared_at': now(), 'user_authorization': 'Delete superseded Qwen proposal/Luna relation-review/Sol adjudication intermediate data after checking current dependencies.',
            'archive_manifest_path': str(ARCHIVE / 'archive_manifest.json'),
            'archive_manifest_sha256': sha(ARCHIVE / 'archive_manifest.json'),
            'targets': [str(p) for p in targets(manifest)], 'payload_file_count': count,
            'payload_bytes': size, 'current_run': str(run), 'current_archive_payload_dependencies': [],
            'preserve_original_archive_manifest': True, 'preserve_compact_exposure_metadata': True,
            'no_worker_or_automation_change': True}
    write(AUDIT / 'deletion_plan.json', plan)
    print(json.dumps({k: plan[k] for k in ['payload_file_count', 'payload_bytes', 'current_run', 'current_archive_payload_dependencies']}), flush=True)


def delete():
    plan = read(AUDIT / 'deletion_plan.json')
    manifest_path = ARCHIVE / 'archive_manifest.json'
    assert sha(manifest_path) == plan['archive_manifest_sha256']
    manifest = read(manifest_path)
    assert [str(p) for p in targets(manifest)] == plan['targets']
    for name, expected in read(AUDIT / 'protected_guards.json').items():
        assert sha(Path(name)) == expected, 'Protected file changed before deletion: ' + name
    active_bindings()
    process = subprocess.run(['ps', '-axo', 'pid=,comm=,args='], capture_output=True, text=True, check=True)
    old_workers = []
    for line in process.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and 'python' in fields[1].lower() and 'pitfalls-cross-lingual' in fields[2]:
            if 'run_full_relation_expansion.py' in fields[2] or 'run_relation_expansion.py propose' in fields[2]:
                old_workers.append(int(fields[0]))
    assert not old_workers, 'Obsolete worker still alive; no deletion performed'
    count, size = verify_payloads(manifest)
    assert count == plan['payload_file_count'] and size == plan['payload_bytes']
    for name in ['historical_model_contacts.jsonl', 'historical_request_metadata.jsonl', 'historical_exposure_summary.json', 'deletion_plan.json']:
        shutil.copyfile(AUDIT / name, ARCHIVE / name)
    started = now()
    write(AUDIT / 'deletion_in_progress.json', {'started_at': started, 'targets': plan['targets']})
    removed = []
    for target in targets(manifest):
        shutil.rmtree(target)
        removed.append(str(target))
        print(json.dumps({'deleted_directory': str(target)}), flush=True)
    for name, expected in read(AUDIT / 'protected_guards.json').items():
        assert sha(Path(name)) == expected, 'Protected file changed after deletion: ' + name
    active_bindings()
    assert all(not Path(name).exists() for name in plan['targets'])
    receipt = {'completed_at': now(), 'started_at': started, 'status': 'obsolete_relation_intermediate_payloads_permanently_deleted',
               'removed_directories': removed, 'removed_payload_files': count, 'removed_logical_bytes': size,
               'protected_current_bindings_raw_ledgers_and_env_unchanged': True,
               'current_pipeline_dependencies_unchanged': True, 'original_archive_manifest_retained_as_historical_inventory': True,
               'compact_historical_exposure_metadata_retained': True, 'old_payloads_are_no_longer_recoverable_from_this_archive': True,
               'worker_or_automation_restarted_or_changed': False}
    write(AUDIT / 'deletion_receipt.json', receipt)
    write(ARCHIVE / 'deletion_receipt.json', receipt)
    (ARCHIVE / 'README.md').write_text('# Obsolete relation-review payloads deleted\n\nThe user authorized permanent deletion on 2026-10-24 after dependency checks. The former primary/ and worktree/ payload directories are gone. archive_manifest.json describes the historical archive at creation time; its recoverable_archive_path fields no longer indicate present/recoverable payloads. See deletion_receipt.json. Only deletion/inventory metadata and compact historical model-contact/request hashes remain. They contain no old Qwen proposals, Luna/Sol semantic judgments, source QA or accepted-candidate datasets. Deletion does not erase exposure and does not establish clean holdout status.\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'delete'])
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else delete()
