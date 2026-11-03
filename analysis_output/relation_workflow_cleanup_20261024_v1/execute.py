from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

AUDIT = Path(__file__).resolve().parent
PROJECT = Path('/Users/qiqing/learn_resource/mutil-language/pitfalls-cross-lingual')
WORKTREE = Path('/Users/qiqing/.codex/worktrees/762b/pitfalls-cross-lingual')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


plan = json.loads((AUDIT / 'cleanup_plan.json').read_text())
targets = {value['path'] for value in plan['targets']}
for value in plan['targets']:
    path = Path(value['path'])
    assert any(path.is_relative_to(root / sub) for root in (PROJECT, WORKTREE)
               for sub in ('scripts', 'configs', 'tests'))
    assert not path.is_symlink() and path.resolve() == path
    assert sha(path) == value['sha256'], str(path)
for value in plan['documentation_edits']:
    path = Path(value['path'])
    assert path.name in {'README.md', 'notes.md', 'task_plan.md', 'task_zh_candidate_pipeline.md'}
    assert not path.is_relative_to(Path(plan['current_run']))
    assert sha(path) == value['before_sha256'], str(path)
for name, expected in plan['protected_guards'].items():
    assert sha(Path(name)) == expected, name

obsolete_names = ['run_full_relation_expansion', 'run_relation_expansion',
                  'relation_expansion_review_gates', 'build_relation_expansion_inventory',
                  'relation_expansion_qwen_luna_v1.json', 'relation_expansion_api.env.example']
remaining_references = []
for root in (PROJECT, WORKTREE):
    for sub in ('scripts', 'tests'):
        for path in (root / sub).glob('*.py'):
            if str(path) not in targets and any(name in path.read_text() for name in obsolete_names):
                remaining_references.append(str(path))
assert not remaining_references, 'Remaining source references: ' + repr(remaining_references)

processes = subprocess.run(['ps', '-axo', 'pid=,comm=,args='], capture_output=True,
                           text=True, check=True)
old_workers = []
for line in processes.stdout.splitlines():
    fields = line.split(None, 2)
    if len(fields) == 3 and 'python' in fields[1].lower() and 'pitfalls-cross-lingual' in fields[2]:
        if 'run_full_relation_expansion.py' in fields[2] or 'run_relation_expansion.py propose' in fields[2]:
            old_workers.append(int(fields[0]))
assert not old_workers, 'Obsolete worker still active; no files removed'

for value in plan['targets']:
    Path(value['path']).unlink()
updated_docs = []
for value in plan['documentation_edits']:
    path = Path(value['path'])
    path.write_text(value['new_text'])
    updated_docs.append({'path': str(path), 'before_sha256': value['before_sha256'],
                         'after_sha256': sha(path)})
for name, expected in plan['protected_guards'].items():
    assert sha(Path(name)) == expected, 'Protected file changed: ' + name
assert all(not Path(name).exists() for name in targets)
receipt = {
    'completed_at': datetime.now(timezone.utc).isoformat(),
    'status': 'superseded_relation_review_workflow_removed',
    'deleted_files': plan['targets'], 'updated_documentation': updated_docs,
    'deleted_dedicated_files': sum(v['kind'] == 'old_dedicated_workflow_file' for v in plan['targets']),
    'deleted_bytecode_caches': sum(v['kind'] == 'old_workflow_bytecode_cache' for v in plan['targets']),
    'current_pipeline_frozen_bindings_unchanged': True, 'shared_api_probe_unchanged': True,
    'env_automation_and_frozen_documentation_unchanged': True,
    'no_old_implementation_backups_created': True, 'no_worker_restart': True,
    'current_source_or_tests_referencing_removed_modules': [],
}
content = json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'
(AUDIT / 'deletion_receipt.json').write_text(content)
archive = WORKTREE / 'analysis_output/_obsolete_archive/relation_review_replaced_20261023'
(archive / 'workflow_files_deletion_receipt.json').write_text(content)
print(json.dumps({key: receipt[key] for key in ['deleted_dedicated_files', 'deleted_bytecode_caches',
                                              'current_pipeline_frozen_bindings_unchanged',
                                              'shared_api_probe_unchanged', 'no_worker_restart']}))
