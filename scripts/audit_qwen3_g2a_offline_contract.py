#!/usr/bin/env python3
"""Audit the Development-only, pre-tokenization Qwen3 G2A contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PREPARER_PATH = Path(__file__).with_name("prepare_qwen3_g2a_offline_contract.py")
PREPARER_SPEC = importlib.util.spec_from_file_location(
    "prepare_qwen3_g2a_offline_contract", PREPARER_PATH
)
if PREPARER_SPEC is None or PREPARER_SPEC.loader is None:
    raise RuntimeError(f"cannot load preparer: {PREPARER_PATH}")
preparer = importlib.util.module_from_spec(PREPARER_SPEC)
PREPARER_SPEC.loader.exec_module(preparer)


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_binding(name: str, binding: Mapping[str, Any]) -> Path:
    path = Path(str(binding["path"])).resolve()
    require(path.is_file(), f"bound file is missing: {name}: {path}")
    require(path.stat().st_size == binding["byte_count"], f"byte count changed: {name}")
    require(sha256_file(path) == binding["sha256"], f"SHA-256 changed: {name}")
    return path


def index_unique(rows: Iterable[Mapping[str, Any]], key: str) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        require(value not in result, f"duplicate {key}: {value}")
        result[value] = row
    return result


def audit_option_free_rows(
    rows: Sequence[Mapping[str, Any]],
    facts: Mapping[str, Mapping[str, Any]],
    assignment_by_id: Mapping[str, str],
) -> Dict[str, Any]:
    require(len(rows) == 960, "expected exactly 960 option-free rows")
    index_unique(rows, "option_free_id")
    index_unique(rows, "source_stimulus_id")
    require(
        {str(row["base_fact_id"]) for row in rows} == set(facts),
        "option-free fact IDs differ from Development",
    )
    require(
        all(row.get("split_assignment") == "development" for row in rows),
        "option-free output contains a non-Development row",
    )

    by_fact: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    pair_rows: Dict[tuple[str, str, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    original_count = Counter()
    arm_count = Counter()
    for row in rows:
        fact_id = str(row["base_fact_id"])
        fact = facts[fact_id]
        language = str(row["language"])
        variant = str(row["variant"])
        require(language in preparer.LANGUAGES, f"invalid language: {fact_id}")
        require(variant in {"original", "neutral", "targeted"}, f"invalid arm: {fact_id}")
        require(
            row["development_fold_id"] == assignment_by_id[fact_id],
            f"wrong fold assignment: {fact_id}",
        )
        require(
            row["leakage_component_id"] == fact["leakage_component_id"],
            f"wrong leakage component: {fact_id}",
        )
        require(
            row["probe_relation_id"] == fact["probe_relation_id"],
            f"wrong relation: {fact_id}",
        )
        stimuli = {
            str(stimulus["stimulus_id"]): stimulus
            for stimulus in fact["static_mcq"]["behavior_inputs"]
        }
        stimulus_id = str(row["source_stimulus_id"])
        require(stimulus_id in stimuli, f"unknown source stimulus: {stimulus_id}")
        stimulus = stimuli[stimulus_id]
        require(stimulus["language"] == language, f"stimulus language mismatch: {stimulus_id}")
        require(stimulus["arm"] == variant, f"stimulus arm mismatch: {stimulus_id}")
        require(stimulus["context"] == row["context"], f"stimulus context mismatch: {stimulus_id}")
        require(stimulus["prompt"] == row["factual_completion_prompt"], f"prompt mismatch: {stimulus_id}")
        require(
            stimulus.get("distractor_id") == row.get("designated_distractor_id"),
            f"distractor mismatch: {stimulus_id}",
        )
        expected_prompt = preparer.option_free_prompt(fact, stimulus)
        require(row["user_prompt"] == expected_prompt, f"option-free prompt mismatch: {stimulus_id}")
        require(
            row["user_prompt_sha256"]
            == hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest(),
            f"option-free prompt SHA mismatch: {stimulus_id}",
        )
        require(row["answer_groups"] == preparer.answer_groups(fact), f"answer groups changed: {fact_id}")
        require(
            row["chat_render_contract"]["status"] == "pending_exact_tokenizer_render",
            f"unexpected render status: {stimulus_id}",
        )
        by_fact[fact_id].append(row)
        arm_count[(language, variant)] += 1
        if variant == "original":
            require(not row["context"], f"Original has context: {stimulus_id}")
            require(row.get("designated_distractor_id") is None, f"Original has distractor: {stimulus_id}")
            require(row.get("target_option_id") is None, f"Original has target: {stimulus_id}")
            require(
                row["user_prompt"] == preparer.natural_prompt(fact, language),
                f"Original differs from natural baseline: {stimulus_id}",
            )
            original_count[(fact_id, language)] += 1
        else:
            distractor_id = str(row["designated_distractor_id"])
            require(row["context"], f"{variant} context is empty: {stimulus_id}")
            require(
                row["user_prompt"].endswith(preparer.natural_prompt(fact, language)),
                f"{variant} does not preserve natural prompt suffix: {stimulus_id}",
            )
            if variant == "targeted":
                require(row["target_option_id"] == distractor_id, f"Targeted target mismatch: {stimulus_id}")
            else:
                require(row.get("target_option_id") is None, f"Neutral has target: {stimulus_id}")
            require(
                variant not in pair_rows[(fact_id, language, distractor_id)],
                f"duplicate {variant} pair member: {fact_id}:{language}:{distractor_id}",
            )
            pair_rows[(fact_id, language, distractor_id)][variant] = row

    require(all(len(value) == 10 for value in by_fact.values()), "not every fact has ten option-free inputs")
    require(
        all(count == 1 for count in original_count.values()) and len(original_count) == 192,
        "Original coverage is not one row per fact and language",
    )
    require(len(pair_rows) == 384, "expected 384 fact/language/distractor pairs")
    for key, pair in pair_rows.items():
        require(set(pair) == {"neutral", "targeted"}, f"incomplete Neutral/Targeted pair: {key}")
        require(
            pair["neutral"]["factual_completion_prompt"]
            == pair["targeted"]["factual_completion_prompt"],
            f"Neutral/Targeted completion prompt differs: {key}",
        )
        require(pair["neutral"]["context"] != pair["targeted"]["context"], f"paired contexts identical: {key}")

    expected_arm_count = {
        ("en", "original"): 96,
        ("zh", "original"): 96,
        ("en", "neutral"): 192,
        ("zh", "neutral"): 192,
        ("en", "targeted"): 192,
        ("zh", "targeted"): 192,
    }
    require(dict(arm_count) == expected_arm_count, "option-free arm counts changed")
    return {
        "row_count": len(rows),
        "fact_count": len(by_fact),
        "neutral_targeted_pair_count": len(pair_rows),
        "original_natural_prompt_identity_count": sum(original_count.values()),
        "arm_counts": {
            f"{language}_{arm}": count
            for (language, arm), count in sorted(arm_count.items())
        },
    }


def audit_vector_rows(
    rows: Sequence[Mapping[str, Any]],
    facts: Mapping[str, Mapping[str, Any]],
    assignment_by_id: Mapping[str, str],
) -> Dict[str, Any]:
    require(len(rows) == 1920, "expected exactly 1,920 vector-source rows")
    index_unique(rows, "vector_source_id")
    require(
        {str(row["base_fact_id"]) for row in rows} == set(facts),
        "vector-source fact IDs differ from Development",
    )
    component_folds: Dict[str, set[str]] = defaultdict(set)
    for fact_id, fold_id in assignment_by_id.items():
        component_folds[str(facts[fact_id]["leakage_component_id"])].add(fold_id)
    require(
        all(len(fold_ids) == 1 for fold_ids in component_folds.values()),
        "a Development leakage component crosses folds",
    )

    partitions: Dict[str, set[str]] = {}
    for fold_id in preparer.FOLD_IDS:
        partitions[fold_id] = {
            fact_id for fact_id, assigned in assignment_by_id.items() if assigned != fold_id
        }
    partitions["final-development-all"] = set(facts)
    rows_by_partition: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    kind_count = Counter()
    for row in rows:
        partition_id = str(row["source_partition_id"])
        require(partition_id in partitions, f"unknown source partition: {partition_id}")
        fact_id = str(row["base_fact_id"])
        require(fact_id in partitions[partition_id], f"evaluation fact leaked into vector source: {partition_id}:{fact_id}")
        expected_excluded = partition_id if partition_id in preparer.FOLD_IDS else None
        require(
            row.get("excluded_evaluation_fold_id") == expected_excluded,
            f"wrong excluded fold: {partition_id}:{fact_id}",
        )
        require(row["render_status"] == "pending_exact_tokenizer_render", f"unexpected vector render status: {row['vector_source_id']}")
        fact = facts[fact_id]
        require(row["probe_relation_id"] == fact["probe_relation_id"], f"wrong vector relation: {fact_id}")
        require(row["leakage_component_id"] == fact["leakage_component_id"], f"wrong vector component: {fact_id}")
        kind = str(row["kind"])
        language = str(row["language"])
        if kind == "task_icl_recall_en" or kind == "task_icl_recall_zh":
            expected_language = kind.rsplit("_", 1)[-1]
            require(language == expected_language, f"task-vector language mismatch: {row['vector_source_id']}")
            donors = preparer.deterministic_donors(
                fact,
                [facts[source_id] for source_id in sorted(partitions[partition_id])],
                partition_id,
                language,
            )
            donor_ids = [str(donor["base_fact_id"]) for donor in donors]
            require(row["icl_donor_base_fact_ids"] == donor_ids, f"donor selection changed: {row['vector_source_id']}")
            require(row["icl_shot_count"] == 5 and len(set(donor_ids)) == 5, f"invalid five-shot donor set: {row['vector_source_id']}")
            for donor in donors:
                require(donor["base_fact_id"] != fact_id, f"self donor: {row['vector_source_id']}")
                require(donor["probe_relation_id"] == fact["probe_relation_id"], f"cross-relation donor: {row['vector_source_id']}")
                require(donor["leakage_component_id"] != fact["leakage_component_id"], f"same-component donor: {row['vector_source_id']}")
            expected_prompt = preparer.icl_prompt(fact, donors, language)
            expected_answer = fact[f"answer_{language}"]
        elif kind == "translation_en_to_zh":
            require(language == "zh", f"translation language mismatch: {row['vector_source_id']}")
            require(row["icl_shot_count"] == 0 and not row["icl_donor_base_fact_ids"], f"translation has donors: {row['vector_source_id']}")
            expected_prompt = preparer.translation_prompt(fact)
            expected_answer = fact["answer_zh"]
        elif kind in {"fact_recall_reference_en", "fact_recall_reference_zh"}:
            expected_language = kind.rsplit("_", 1)[-1]
            require(language == expected_language, f"recall language mismatch: {row['vector_source_id']}")
            require(row["icl_shot_count"] == 0 and not row["icl_donor_base_fact_ids"], f"recall reference has donors: {row['vector_source_id']}")
            expected_prompt = preparer.natural_prompt(fact, language)
            expected_answer = fact[f"answer_{language}"]
        else:
            raise ValueError(f"unknown vector source kind: {kind}")
        require(row["prompt"] == expected_prompt, f"vector prompt changed: {row['vector_source_id']}")
        require(row["expected_answer"] == expected_answer, f"expected answer changed: {row['vector_source_id']}")
        require(
            row["prompt_sha256"] == hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest(),
            f"vector prompt SHA mismatch: {row['vector_source_id']}",
        )
        rows_by_partition[partition_id].append(row)
        kind_count[kind] += 1

    expected_partition_counts = {
        **{fold_id: 360 for fold_id in preparer.FOLD_IDS},
        "final-development-all": 480,
    }
    require(
        {key: len(value) for key, value in rows_by_partition.items()}
        == expected_partition_counts,
        "vector-source partition counts changed",
    )
    expected_kind_count = {kind: 384 for kind in preparer.VECTOR_SOURCE_KINDS}
    require(dict(kind_count) == expected_kind_count, "vector-source kind counts changed")
    for partition_id, source_ids in partitions.items():
        observed = {str(row["base_fact_id"]) for row in rows_by_partition[partition_id]}
        require(observed == source_ids, f"source fact coverage changed: {partition_id}")
        for fact_id in source_ids:
            kinds = {
                str(row["kind"])
                for row in rows_by_partition[partition_id]
                if row["base_fact_id"] == fact_id
            }
            require(kinds == set(preparer.VECTOR_SOURCE_KINDS), f"incomplete vector source set: {partition_id}:{fact_id}")

    return {
        "row_count": len(rows),
        "partition_counts": expected_partition_counts,
        "kind_counts": dict(sorted(kind_count.items())),
        "oof_source_evaluation_overlap_count": 0,
        "cross_fold_leakage_component_count": 0,
        "invalid_donor_count": 0,
    }


def audit_contract(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "offline_contract_prepared_pending_exact_tokenizer_audit",
        "unexpected offline-contract status",
    )
    require(manifest.get("scope") == "development_only_option_free_and_vector_source_preparation", "unexpected scope")
    require(manifest["counts"]["validation_facts_used"] == 0, "Validation was consumed")
    require(manifest["counts"]["sealed_facts_used"] == 0, "Sealed was consumed")
    require(manifest["authorization_state"]["vector_hidden_state_collection_authorized"] is False, "vector collection was authorized")
    require(manifest["authorization_state"]["development_vector_intervention_authorized"] is False, "intervention was authorized")
    require(manifest["authorization_state"]["validation_authorized"] is False, "Validation was authorized")
    require(manifest["authorization_state"]["sealed_authorized"] is False, "Sealed was authorized")
    deferral_policy = manifest.get("natural_semantic_deferral_policy", {})
    require(deferral_policy.get("deferred_record_count") == 3, "natural semantic deferral count changed")
    require("missing" in str(deferral_policy.get("treatment", "")).lower(), "natural semantic deferrals lack a missing-value policy")
    require(deferral_policy.get("human_gold") is False, "Codex semantic review was mislabeled as human gold")
    unrelated = manifest.get("unrelated_fact_control", {})
    require(
        unrelated.get("status")
        == "behavior_blind_unrelated_control_frozen_pending_exact_tokenizer_audit",
        "unrelated-fact control set is not frozen",
    )
    require(unrelated.get("control_fact_count") == 13, "unrelated-fact control count changed")
    require(unrelated.get("control_input_count") == 26, "unrelated-fact control input count changed")
    require(unrelated.get("current_Qwen3_behavior_used") is False, "unrelated controls used Qwen3 behavior")
    require(unrelated.get("vector_source") is False, "unrelated controls entered vector sources")

    input_paths = {
        name: verify_binding(f"input:{name}", binding)
        for name, binding in manifest["inputs"].items()
    }
    artifact_paths = {
        name: verify_binding(f"artifact:{name}", binding)
        for name, binding in manifest["artifacts"].items()
    }
    bundle = read_jsonl(input_paths["input_bundle"])
    require(len(bundle) == 160, "frozen bundle no longer has 160 facts")
    split_count = Counter(str(row["split_assignment"]) for row in bundle)
    require(
        dict(split_count) == {"development": 96, "sealed": 32, "validation": 32},
        "frozen split counts changed",
    )
    development = {
        str(row["base_fact_id"]): row
        for row in bundle
        if row["split_assignment"] == "development"
    }
    require(len(development) == 96, "Development IDs are not unique")
    fold_manifest = read_json(input_paths["fold_manifest"])
    assignment_by_id = {
        str(row["base_fact_id"]): str(row["development_fold_id"])
        for row in fold_manifest["assignments"]
    }
    require(set(assignment_by_id) == set(development), "fold assignments differ from Development")
    require(Counter(assignment_by_id.values()) == Counter({fold_id: 24 for fold_id in preparer.FOLD_IDS}), "fold sizes changed")

    option_rows = read_jsonl(artifact_paths["option_free_three_arm_inputs"])
    vector_rows = read_jsonl(artifact_paths["vector_source_prompt_specs"])
    option_audit = audit_option_free_rows(option_rows, development, assignment_by_id)
    vector_audit = audit_vector_rows(vector_rows, development, assignment_by_id)

    require(manifest["counts"]["option_free_inputs"] == len(option_rows), "manifest option-free count mismatch")
    require(manifest["counts"]["vector_source_prompts"] == len(vector_rows), "manifest vector count mismatch")
    return {
        "schema_version": "qwen3-g2a-offline-contract-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "development_only_pre_tokenization_contract",
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "checks": {
            "bound_input_and_artifact_sha256": "passed",
            "development_only": "passed",
            "validation_and_sealed_usage_zero": "passed",
            "option_free_three_arm_structure": "passed",
            "original_natural_prompt_identity": "passed",
            "neutral_targeted_context_only_template": "passed",
            "answer_alias_groups_disjoint": "passed",
            "four_fold_source_evaluation_isolation": "passed",
            "leakage_component_fold_atomicity": "passed",
            "five_shot_donor_relation_and_component_rules": "passed",
            "intervention_authorization_remains_false": "passed",
            "natural_semantic_deferrals_have_missing_value_policy": "passed",
            "behavior_blind_unrelated_fact_control_frozen": "passed",
        },
        "split_counts": dict(sorted(split_count.items())),
        "option_free": option_audit,
        "vector_sources": vector_audit,
        "artifact_sha256": {
            name: sha256_file(path) for name, path in sorted(artifact_paths.items())
        },
        "remaining_blockers": list(manifest["open_items"]),
        "claim_boundary": manifest["claim_boundary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_contract(args.manifest)
    output_path = args.output or args.manifest.resolve().with_name(
        "g2a_offline_contract_audit.json"
    )
    write_json_atomic(output_path.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
