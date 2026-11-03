"""Run on the remote host; audit saved records without inference or selection code."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/autodl-tmp/pnt-open-development-20261016')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


base = {r['option_free_id']: r for r in rows(ROOT/'results/open_results_labeled.jsonl')}
assignments = read(ROOT/'inputs/development_fold_manifest.json')['assignments']
folds = {r['base_fact_id']: r['development_fold_id'] for r in assignments}
assert len(folds) == 96
lookup = {(r['base_fact_id'], r['language'], r['variant'], r['designated_distractor_id']): r for r in base.values()}
resistant = {}
for row in rows(ROOT/'results/open_variant_labels.jsonl'):
    triple = [lookup[(row['base_fact_id'], 'zh', arm, None if arm == 'original' else row['distractor_id'])]
              for arm in ('original', 'neutral', 'targeted')]
    if all(r['generation_correct'] is True for r in triple):
        resistant.setdefault(row['base_fact_id'], set()).update(r['option_free_id'] for r in triple)
assert len(resistant) == 14
checks = {}
bound_files = [ROOT/'repair_config_v3.json', ROOT/'repair_config_v4.json', ROOT/'audit_repair_closure.py',
               ROOT/'inputs/development_fold_manifest.json', ROOT/'results/open_results_labeled.jsonl',
               ROOT/'results/open_variant_labels.jsonl', ROOT/'results/unrelated_control_semantic_review.jsonl']
total_calls = 0
for version, expected_candidates in [('repair-v3', 3), ('repair-v4', 5)]:
    out = ROOT/version
    config_path = ROOT/('repair_config_'+version.split('-')[-1]+'.json')
    manifest = read(out/'oof_vector_manifest.json')
    assert sha(out/'oof_vectors.pt') == manifest['vectors_sha256']
    assert manifest['source_count'] == 1440
    assert manifest['validation_facts_used'] == manifest['sealed_facts_used'] == 0
    for partition, info in manifest['partitions'].items():
        expected = {fid for fid, fold in folds.items() if fold != partition}
        assert info['source_facts'] == 72 and set(info['source_fact_ids']) == expected
    noop = read(out/'no_op_injection_audit.json')
    assert noop['status'] == 'passed' and len(noop['checks']) == 4
    assert all(r['exact_generation_match'] and r['absolute_margin_delta'] < 1e-5 for r in noop['checks'])
    identity = read(out/'generate-identity.json')
    assert identity['config_sha256'] == sha(config_path)
    assert identity['common_code_sha256'] == sha(ROOT/'scripts/qwen3_open_repair_common.py')
    decisions = {r['review_key']: r for r in rows(out/'semantic_review_decisions.jsonl')}
    specs = read(out/'coarse_selection.json')['candidates']
    assert len(specs) == expected_candidates
    summaries = read(out/'generation_metrics.json')
    candidates = {}
    for spec in specs:
        path = out/('generation-'+spec['id']+'.jsonl')
        raw = rows(path)
        labeled = rows(out/'labeled'/path.name)
        assert len(raw) == len(labeled) == 532
        assert len({(r['input_id'], r['evaluation_partition']) for r in raw}) == 532
        assert sum(r['base_fact_id'] is not None for r in raw) == 480
        assert {r['base_fact_id'] for r in raw if r['base_fact_id']} == set(folds)
        assert all(r['language'] == 'zh' for r in raw)
        assert all(folds[r['base_fact_id']] == r['evaluation_partition'] for r in raw if r['base_fact_id'])
        for before, after in zip(raw, labeled):
            assert before['input_id'] == after['input_id'] and before['generation'] == after['generation']
            decision = decisions.get(before['review_key'])
            if before['generation_correct'] is None and decision:
                assert decision['human_gold'] is False and decision['reviewer_type'] == 'codex_proxy'
                assert decision['generation_sha256'] == hashlib.sha256(before['generation'].encode()).hexdigest()
                assert after['generation_correct'] == decision['semantic_correct']
            else:
                assert after['generation_correct'] == before['generation_correct']
        by_id = {r['input_id']: r for r in labeled if r['base_fact_id']}
        harmed = sorted(fid for fid, ids in resistant.items() if any(by_id[key]['generation_correct'] is False for key in ids))
        assert harmed == sorted(summaries[spec['id']]['resistant_harmed'])
        candidates[spec['id']] = {'calls': len(raw), 'known_resistant_harmed_facts': harmed,
                                  'unresolved_primary_pairs': summaries[spec['id']]['unresolved_paired_inputs'],
                                  'eligible': not summaries[spec['id']]['ineligibility_reasons']}
        assert not candidates[spec['id']]['eligible']
        total_calls += len(raw)
        bound_files.extend([path, out/'labeled'/path.name])
    selection = read(out/'generation_selection.json')
    assert not selection['eligible_specs'] and selection['validation_used'] == selection['sealed_used'] == 0
    assert not (out/'repair_rule_frozen.json').exists() and not (out/'final_vectors.pt').exists()
    assert not list(out.glob('generation-random-*.jsonl'))
    checks[version] = {'candidates': candidates, 'selection_status': selection['status'],
                       'oof_source_isolation': 'passed', 'no_op_injection': 'passed'}
    bound_files.extend(out/name for name in ('oof_vector_manifest.json', 'coarse_selection.json',
        'generation_metrics.json', 'generation_selection.json', 'semantic_review_decisions.jsonl',
        'review_export.jsonl', 'no_op_injection_audit.json', 'generate-identity.json', 'open_trajectory_summary.json'))
opened = list(ROOT.glob('**/opened.json'))
assert not opened
bound_files.extend(sorted((ROOT/'scripts').glob('*qwen3*.py')))
audit = {'status': 'passed', 'audit_scope': 'independent saved-artifact integrity checks; not independent holdout confirmation',
         'created_at': datetime.now(timezone.utc).isoformat(), 'candidate_generation_calls': total_calls,
         'checks': checks, 'holdout_opened_records': [], 'validation_used': 0, 'sealed_used': 0,
         'final_repair_rule_frozen': False, 'final_source_vectors_collected': False,
         'bound_files': {str(p.relative_to(ROOT)): sha(p) for p in bound_files}}
destination = ROOT/'repair_closure_audit.json'
destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
decision = {'status': 'development_concluded_without_confirmable_repair_rule',
            'latest_gate_status': checks['repair-v4']['selection_status'],
            'v3_candidates': 3, 'v4_candidates': 5, 'eligible_candidates': 0,
            'random_comparison': 'not_started_no_generation_eligible_candidate',
            'final_repair_rule_frozen': False, 'independent_confirmation': 'not_authorized_by_G2_evidence',
            'validation_used': 0, 'sealed_used': 0, 'closure_audit_sha256': sha(destination),
            'next_required_evidence': 'A new prospectively specified Development-only repair method that passes the unchanged retention conditions; the completed finite candidate pool cannot advance. A passing candidate must then clear five random controls before full-Development vector freeze, Validation then Sealed.'}
(ROOT/'repair_final_decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'audit': audit['status'], 'candidate_calls': total_calls, 'decision': decision['status']}, ensure_ascii=False))
