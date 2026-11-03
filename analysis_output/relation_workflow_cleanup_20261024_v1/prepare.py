from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

PROJECT = Path('/Users/qiqing/learn_resource/mutil-language/pitfalls-cross-lingual')
WORKTREE = Path('/Users/qiqing/.codex/worktrees/762b/pitfalls-cross-lingual')
AUDIT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


relative_paths = [
    'scripts/build_relation_expansion_inventory.py',
    'scripts/relation_expansion_review_gates.py',
    'scripts/run_full_relation_expansion.py',
    'scripts/run_relation_expansion.py',
    'configs/relation_expansion_qwen_luna_v1.json',
    'configs/relation_expansion_api.env.example',
    'tests/test_full_relation_expansion.py',
    'tests/test_relation_expansion_inventory.py',
    'tests/test_relation_expansion_review_gates.py',
    'tests/test_relation_expansion_workflow.py',
]
targets = []
for root in (PROJECT, WORKTREE):
    for relative in relative_paths:
        path = root / relative
        assert path.is_file() and not path.is_symlink()
        targets.append({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
                        'kind': 'old_dedicated_workflow_file'})
    for sub in ('scripts', 'tests'):
        stems = [Path(value).stem for value in relative_paths if value.startswith(sub + '/')]
        for path in sorted((root / sub / '__pycache__').glob('*.pyc')):
            if any(path.name.startswith(stem + '.') for stem in stems):
                targets.append({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
                                'kind': 'old_workflow_bytecode_cache'})

docs = []
for root in (PROJECT, WORKTREE):
    names = ['notes.md', 'task_plan.md'] + (['tasks/README.md'] if root == WORKTREE else [])
    for name in names:
        path = root / name
        text = path.read_text()
        assert text.startswith('## 2026-10-23')
        parts = text.split('\n\n', 2)
        if name == 'tasks/README.md':
            link = 'task_zh_candidate_pipeline.md'
        elif root == WORKTREE:
            link = 'tasks/task_zh_candidate_pipeline.md'
        else:
            link = str(WORKTREE / 'tasks/task_zh_candidate_pipeline.md')
        paragraph = (f'以[中文候选流水线]({link})及 analysis_output/active_data_pipeline.json 为准：'
                     'Qwen规范化英文并翻译中文，Luna只审核翻译，Sol只裁决翻译疑难项。'
                     '旧关系逐条复审方案、专用运行器、配置、测试及派生数据已删除；'
                     '仅保留删除清单和历史曝光元数据。原始抽取账本、历史译文、160实验及'
                     '当前7,250条来源输入保留。中文通过不自动满足正式实验/PNT冻结。')
        docs.append({'path': str(path), 'before_sha256': sha(path),
                     'new_text': '## 2026-10-24 当前数据建设方案\n\n' + paragraph + '\n\n' + parts[2]})

path = WORKTREE / 'tasks/task_zh_candidate_pipeline.md'
text = path.read_text().replace(
    '生效：2026-10-23，取代此前 relation_expansion 系列“逐条关系 Luna/Sol 审核”的方案。',
    '生效：2026-10-23。当前数据候选建设仅采用本方案。')
old = ('旧 relation_expansion 方案及最终数据已移出原活动目录，在analysis_output/_obsolete_archive/'
       'relation_review_replaced_20261023保留可恢复归档和逐文件哈希。原始抽取账本、'
       '原中文/五语译文和原160实验未删除；历史曝光也没有清零。')
new = ('已废止的关系逐条复审方案、专用实现/配置/测试及派生数据已于2026-10-24删除。'
       '原归档位置仅保留删除凭据、逐文件哈希清单与历史模型接触元数据，'
       '不再保留可恢复的旧方案或结果。原始抽取账本、原中文/五语译文和原160实验保留；历史曝光没有清零。')
assert old in text
text = text.replace(old, new).replace(
    '原relation_expansion系列命令与其Luna/Sol关系审核合同属于已废止历史路径，不用于本方案。', '')
docs.append({'path': str(path), 'before_sha256': sha(path), 'new_text': text})

pointer_path = PROJECT / 'analysis_output/active_data_pipeline.json'
pointer = json.loads(pointer_path.read_text())
run = Path(pointer['run_dir'])
manifest = json.loads((run / 'manifest.json').read_text())
assert sha(run / 'manifest.json') == pointer['manifest_sha256']
guards = dict(manifest['bindings'])
for path in [pointer_path, run / 'manifest.json', run / 'protocol.md',
             run / 'documentation_history_manifest.json', PROJECT / 'scripts/probe_relation_expansion_api.py',
             WORKTREE / 'scripts/probe_relation_expansion_api.py', PROJECT / 'utils/.env',
             Path('/Users/qiqing/.codex/automations/automation/automation.toml')]:
    guards[str(path)] = sha(path)
for value in json.loads((run / 'documentation_history_manifest.json').read_text())['snapshots'].values():
    guards[value['snapshot']] = value['sha256']
for name, expected in guards.items():
    assert sha(Path(name)) == expected
assert not ({value['path'] for value in targets} & set(guards))
plan = {'created_at': datetime.now(timezone.utc).isoformat(),
        'user_authorization': 'Delete the superseded Qwen conversion/Luna relation-review/Sol adjudication scheme and related files.',
        'targets': targets, 'documentation_edits': docs, 'protected_guards': guards,
        'current_run': str(run), 'shared_api_probe_preserved': True,
        'no_worker_automation_or_frozen_runtime_change': True}
(AUDIT / 'cleanup_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'dedicated_files': sum(value['kind'] == 'old_dedicated_workflow_file' for value in targets),
                  'bytecode_cache_files': sum(value['kind'] == 'old_workflow_bytecode_cache' for value in targets),
                  'documentation_files': len(docs),
                  'current_tests': [value.name for value in (WORKTREE / 'tests').glob('test_zh_candidate*.py')]}))
