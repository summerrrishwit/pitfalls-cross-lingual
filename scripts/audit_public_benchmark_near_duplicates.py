#!/usr/bin/env python3
"""Deterministic lexical candidate audit for public-benchmark fact bundles.

This script is deliberately an offline *candidate generator*.  It detects
normalised answer-alias leakage across provisional splits and emits exact or
near-lexical question/fact pairs for later review.  It does not make semantic
duplicate decisions and must not be used as evidence that semantic review is
complete.

Candidate generation is sub-quadratic by construction: exact values and
aliases use inverted indexes, while near matches use deterministic MinHash
bands with a bounded lexical-neighbour window per band.  No model, API, or
network access is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


SUMMARY_SCHEMA_VERSION = "public-benchmark-lexical-candidate-audit-v2"
PAIR_SCHEMA_VERSION = "public-benchmark-lexical-candidate-pair-v2"
INPUT_BUNDLE_SCHEMA_VERSION = "factual-perturbation-input-bundle-v1"
TEXT_NORMALIZATION_VERSION = "nfkc-casefold-unicode-word-v1"
ANSWER_NORMALIZATION_VERSION = "nfkc-casefold-whitespace-v1"
BLOCKING_VERSION = "token-minhash-16x2-lexical-window-v1"
ALLOWED_SPLITS = {"development", "validation", "sealed"}
DEFAULT_QUESTION_THRESHOLD = 0.80
DEFAULT_FACT_THRESHOLD = 0.80
DEFAULT_MAX_BUCKET_NEIGHBORS = 24
DEFAULT_MAX_EXAMPLES = 20
MINHASH_COMPONENTS = 16
MINHASH_BAND_ROWS = 2
TEXT_EVIDENCE_TYPES = frozenset(
    {
        "question_exact",
        "question_near_lexical",
        "canonical_fact_exact",
        "canonical_fact_near_lexical",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_lexical(value: Any) -> str:
    """Return a conservative punctuation-insensitive lexical normalisation."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    return " ".join(tokens)


def normalize_answer(value: Any) -> str:
    """Match the upstream provisional split's conservative answer key."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"\s+", " ", text)


def unique_strings(values: Iterable[Any]) -> Tuple[str, ...]:
    output: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        raw = value.strip()
        key = normalize_answer(raw)
        if raw and key and key not in seen:
            seen.add(key)
            output.append(raw)
    return tuple(output)


def first_string(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_aliases(record: Mapping[str, Any], answer: str) -> Tuple[str, ...]:
    values: List[Any] = [answer]
    for key in ("answer_aliases_en", "answer_aliases"):
        aliases = record.get(key)
        if isinstance(aliases, list):
            values.extend(aliases)
    source_answer = record.get("source_answer")
    if isinstance(source_answer, str):
        values.append(source_answer)
    snapshot = record.get("source_snapshot")
    if isinstance(snapshot, dict):
        source_record = snapshot.get("record")
        if isinstance(source_record, dict):
            metadata = source_record.get("upstream_metadata")
            if isinstance(metadata, dict) and isinstance(metadata.get("answer_aliases"), list):
                values.extend(metadata["answer_aliases"])
    return unique_strings(values)


def read_jsonl(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: List[Tuple[int, Dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append((line_number, value))
    if not records:
        raise ValueError(f"Input JSONL is empty: {path}")
    return records


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record).decode("utf-8"))
            handle.write("\n")
    temporary.replace(path)


@dataclass(frozen=True)
class AuditRecord:
    role: str
    line_number: int
    input_path: str
    input_sha256: str
    record_sha256: str
    audit_record_id: str
    raw: Mapping[str, Any]
    question: str
    question_normalized: str
    canonical_fact: str
    canonical_fact_normalized: str
    answer: str
    answer_normalized: str
    answer_aliases: Tuple[str, ...]
    answer_aliases_normalized: Tuple[str, ...]
    split_assignment: Optional[str]
    split_group_id: Optional[str]

    @property
    def sort_key(self) -> Tuple[str, str, int]:
        return (self.role, self.audit_record_id, self.line_number)


def make_audit_record(
    role: str,
    line_number: int,
    raw: Mapping[str, Any],
    input_path: Path,
    input_sha256: str,
) -> AuditRecord:
    if role == "current_bundle" and raw.get("schema_version") != INPUT_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported bundle schema at {input_path}:{line_number}: "
            f"{raw.get('schema_version')!r}"
        )
    if role == "current_bundle" and not str(raw.get("base_fact_id") or "").strip():
        raise ValueError(f"Missing base_fact_id at {input_path}:{line_number}")

    question = first_string(raw, ("source_question_en", "source_question", "question"))
    canonical_fact = first_string(raw, ("canonical_fact_en", "canonical_fact"))
    answer = first_string(raw, ("answer_en", "answer", "source_answer"))
    aliases = extract_aliases(raw, answer)
    record_sha256 = sha256_value(raw)
    identity = {
        "role": role,
        "base_fact_id": raw.get("base_fact_id"),
        "candidate_id": raw.get("candidate_id"),
        "source_id": raw.get("source_id"),
        "record_sha256": record_sha256,
        "line_number": line_number,
    }
    audit_record_id = f"audit_rec_{sha256_value(identity)[:24]}"
    split = raw.get("split_assignment")
    split_group = raw.get("split_group_id")
    return AuditRecord(
        role=role,
        line_number=line_number,
        input_path=str(input_path.resolve()),
        input_sha256=input_sha256,
        record_sha256=record_sha256,
        audit_record_id=audit_record_id,
        raw=raw,
        question=question,
        question_normalized=normalize_lexical(question),
        canonical_fact=canonical_fact,
        canonical_fact_normalized=normalize_lexical(canonical_fact),
        answer=answer,
        answer_normalized=normalize_answer(answer),
        answer_aliases=aliases,
        answer_aliases_normalized=tuple(
            sorted({normalize_answer(value) for value in aliases if normalize_answer(value)})
        ),
        split_assignment=str(split) if split is not None else None,
        split_group_id=str(split_group) if split_group is not None else None,
    )


def load_records(path: Path, role: str) -> List[AuditRecord]:
    input_sha256 = sha256_file(path)
    return [
        make_audit_record(role, line_number, raw, path, input_sha256)
        for line_number, raw in read_jsonl(path)
    ]


def record_provenance(record: AuditRecord) -> Dict[str, Any]:
    raw = record.raw
    return {
        "audit_record_id": record.audit_record_id,
        "input_role": record.role,
        "input_path": record.input_path,
        "input_sha256": record.input_sha256,
        "input_line_number": record.line_number,
        "input_record_sha256": record.record_sha256,
        "base_fact_id": raw.get("base_fact_id"),
        "candidate_id": raw.get("candidate_id"),
        "source_id": raw.get("source_id"),
        "source_dataset": raw.get("source_dataset"),
        "source_subset": raw.get("source_subset"),
        "source_path": raw.get("source_path"),
        "source_original_index": raw.get("source_original_index"),
        "upstream_provenance": raw.get("provenance"),
    }


def pair_side(record: AuditRecord) -> Dict[str, Any]:
    return {
        "provenance": record_provenance(record),
        "split_assignment": record.split_assignment,
        "split_group_id": record.split_group_id,
        "question_raw": record.question,
        "canonical_fact_raw": record.canonical_fact,
        "answer_raw": record.answer,
        "answer_aliases_raw": list(record.answer_aliases),
    }


def build_review_contract(match_types: Sequence[str]) -> Dict[str, Any]:
    """Return the adjudication fields required by this pair's evidence."""

    required: List[str] = []
    if "answer_alias_exact" in match_types:
        required.extend(("alias_valid", "same_answer_entity"))
    if any(match_type in TEXT_EVIDENCE_TYPES for match_type in match_types):
        required.append("semantic_duplicate")
    if not required:
        raise ValueError(f"No review contract for match types: {list(match_types)!r}")
    return {
        "status": "pending_adjudication",
        "required_adjudications": required,
        **{field: None for field in required},
    }


class PairCollector:
    def __init__(self) -> None:
        self._pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add(
        self,
        first: AuditRecord,
        second: AuditRecord,
        match_type: str,
        evidence: Mapping[str, Any],
    ) -> None:
        if first.audit_record_id == second.audit_record_id:
            return
        if first.role == "comparison_canonical" and second.role == "comparison_canonical":
            return
        left, right = sorted((first, second), key=lambda item: item.sort_key)
        normalized_evidence = dict(evidence)
        if left.audit_record_id != first.audit_record_id:
            left_text = normalized_evidence.get("left_normalized_text")
            right_text = normalized_evidence.get("right_normalized_text")
            if left_text is not None and right_text is not None:
                normalized_evidence["left_normalized_text"] = right_text
                normalized_evidence["right_normalized_text"] = left_text
        key = (left.audit_record_id, right.audit_record_id)
        entry = self._pairs.setdefault(
            key,
            {"left": left, "right": right, "evidence": {}},
        )
        if match_type == "answer_alias_exact":
            aliases = entry["evidence"].setdefault(match_type, {})
            normalized = str(normalized_evidence["normalized_alias"])
            aliases[normalized] = normalized_evidence
        else:
            entry["evidence"][match_type] = normalized_evidence

    def finalize(self) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for key in sorted(self._pairs):
            entry = self._pairs[key]
            left: AuditRecord = entry["left"]
            right: AuditRecord = entry["right"]
            serialized_evidence: Dict[str, Any] = {}
            for match_type in sorted(entry["evidence"]):
                value = entry["evidence"][match_type]
                if match_type == "answer_alias_exact":
                    serialized_evidence[match_type] = [value[name] for name in sorted(value)]
                else:
                    serialized_evidence[match_type] = value
            match_types = sorted(serialized_evidence)
            scope = (
                "within_current_bundle"
                if left.role == right.role == "current_bundle"
                else "current_vs_comparison_canonical"
            )
            pair_hash_payload = {
                "left": left.audit_record_id,
                "right": right.audit_record_id,
                "match_types": match_types,
            }
            output.append(
                {
                    "schema_version": PAIR_SCHEMA_VERSION,
                    "pair_id": f"lexpair_{sha256_value(pair_hash_payload)[:24]}",
                    "audit_scope": scope,
                    "match_types": match_types,
                    "match_evidence": serialized_evidence,
                    "cross_split": (
                        scope == "within_current_bundle"
                        and left.split_assignment in ALLOWED_SPLITS
                        and right.split_assignment in ALLOWED_SPLITS
                        and left.split_assignment != right.split_assignment
                    ),
                    "left": pair_side(left),
                    "right": pair_side(right),
                    "review": build_review_contract(match_types),
                }
            )
        output.sort(key=lambda item: str(item["pair_id"]))
        return output


def spanning_pairs(records: Sequence[AuditRecord]) -> Iterable[Tuple[AuditRecord, AuditRecord]]:
    """Yield O(n) useful edges and never comparison-vs-comparison edges."""

    current = sorted(
        (item for item in records if item.role == "current_bundle"),
        key=lambda item: item.sort_key,
    )
    comparison = sorted(
        (item for item in records if item.role == "comparison_canonical"),
        key=lambda item: item.sort_key,
    )
    for index in range(1, len(current)):
        yield current[index - 1], current[index]
    if current:
        for item in comparison:
            yield current[0], item


def exact_field_candidates(
    records: Sequence[AuditRecord],
    field_name: str,
    normalized_attribute: str,
    collector: PairCollector,
) -> Dict[str, Any]:
    groups: Dict[str, List[AuditRecord]] = defaultdict(list)
    for record in records:
        value = str(getattr(record, normalized_attribute))
        if value:
            groups[value].append(record)
    matched_groups = 0
    for normalized, members in sorted(groups.items()):
        if len(members) < 2 or not any(item.role == "current_bundle" for item in members):
            continue
        matched_groups += 1
        for first, second in spanning_pairs(members):
            collector.add(
                first,
                second,
                f"{field_name}_exact",
                {
                    "normalization_version": TEXT_NORMALIZATION_VERSION,
                    "normalized_text": normalized,
                    "normalized_text_sha256": hashlib.sha256(
                        normalized.encode("utf-8")
                    ).hexdigest(),
                },
            )
    return {
        "nonempty_normalized_value_count": len(groups),
        "exact_matched_group_count": matched_groups,
    }


@lru_cache(maxsize=None)
def token_set(text: str) -> FrozenSet[str]:
    return frozenset(text.split())


@lru_cache(maxsize=None)
def char_ngrams(text: str, size: int = 3) -> FrozenSet[str]:
    compact = f"  {text}  "
    if len(compact) <= size:
        return frozenset({compact})
    return frozenset(
        compact[index : index + size] for index in range(len(compact) - size + 1)
    )


def jaccard(first: FrozenSet[str], second: FrozenSet[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def lexical_similarity(first: str, second: str) -> Dict[str, float]:
    first_tokens = token_set(first)
    second_tokens = token_set(second)
    intersection = len(first_tokens & second_tokens)
    minimum = min(len(first_tokens), len(second_tokens))
    token_containment = intersection / minimum if minimum else 0.0
    token_jaccard = jaccard(first_tokens, second_tokens)
    character_jaccard = jaccard(char_ngrams(first), char_ngrams(second))
    length_ratio = (
        min(len(first), len(second)) / max(len(first), len(second))
        if first and second
        else 0.0
    )
    return {
        "token_jaccard": round(token_jaccard, 6),
        "token_containment": round(token_containment, 6),
        "character_trigram_jaccard": round(character_jaccard, 6),
        "length_ratio": round(length_ratio, 6),
        "decision_score": round(max(token_jaccard, character_jaccard), 6),
    }


@lru_cache(maxsize=None)
def stable_u64(component: int, token: str) -> int:
    payload = f"{component}\u241f{token}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def minhash_signature(tokens: FrozenSet[str]) -> Tuple[int, ...]:
    return tuple(
        min(stable_u64(component, token) for token in tokens)
        for component in range(MINHASH_COMPONENTS)
    )


def choose_current_pair(
    first: Sequence[AuditRecord], second: Sequence[AuditRecord]
) -> Optional[Tuple[AuditRecord, AuditRecord]]:
    left = sorted(
        (item for item in first if item.role == "current_bundle"),
        key=lambda item: item.sort_key,
    )
    right = sorted(
        (item for item in second if item.role == "current_bundle"),
        key=lambda item: item.sort_key,
    )
    if not left or not right:
        return None
    for left_item in left:
        for right_item in right:
            if (
                left_item.split_assignment in ALLOWED_SPLITS
                and right_item.split_assignment in ALLOWED_SPLITS
                and left_item.split_assignment != right_item.split_assignment
            ):
                return left_item, right_item
    return left[0], right[0]


def representative_pairs(
    first: Sequence[AuditRecord], second: Sequence[AuditRecord]
) -> Iterable[Tuple[AuditRecord, AuditRecord]]:
    current_pair = choose_current_pair(first, second)
    if current_pair is not None:
        yield current_pair
    first_current = sorted(
        (item for item in first if item.role == "current_bundle"), key=lambda item: item.sort_key
    )
    first_comparison = sorted(
        (item for item in first if item.role == "comparison_canonical"),
        key=lambda item: item.sort_key,
    )
    second_current = sorted(
        (item for item in second if item.role == "current_bundle"), key=lambda item: item.sort_key
    )
    second_comparison = sorted(
        (item for item in second if item.role == "comparison_canonical"),
        key=lambda item: item.sort_key,
    )
    if first_current and second_comparison:
        yield first_current[0], second_comparison[0]
    if second_current and first_comparison:
        yield second_current[0], first_comparison[0]


def near_field_candidates(
    records: Sequence[AuditRecord],
    field_name: str,
    normalized_attribute: str,
    threshold: float,
    max_bucket_neighbors: int,
    collector: PairCollector,
) -> Dict[str, Any]:
    groups: Dict[str, List[AuditRecord]] = defaultdict(list)
    for record in records:
        value = str(getattr(record, normalized_attribute))
        if value:
            groups[value].append(record)

    eligible_texts = sorted(text for text in groups if len(token_set(text)) >= 3)
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[str]] = defaultdict(list)
    for text in eligible_texts:
        signature = minhash_signature(token_set(text))
        for band_start in range(0, MINHASH_COMPONENTS, MINHASH_BAND_ROWS):
            band = signature[band_start : band_start + MINHASH_BAND_ROWS]
            buckets[(band_start // MINHASH_BAND_ROWS, band)].append(text)

    candidate_text_pairs: Set[Tuple[str, str]] = set()
    truncated_bucket_count = 0
    skipped_bucket_pair_count = 0
    for bucket_key in sorted(buckets):
        texts = sorted(set(buckets[bucket_key]))
        if len(texts) > max_bucket_neighbors + 1:
            truncated_bucket_count += 1
            total = len(texts) * (len(texts) - 1) // 2
            kept = sum(
                min(max_bucket_neighbors, len(texts) - index - 1)
                for index in range(len(texts))
            )
            skipped_bucket_pair_count += total - kept
        for index, first in enumerate(texts):
            stop = min(len(texts), index + max_bucket_neighbors + 1)
            for second in texts[index + 1 : stop]:
                candidate_text_pairs.add((first, second))

    qualified_text_pair_count = 0
    emitted_record_pair_count = 0
    for first, second in sorted(candidate_text_pairs):
        metrics = lexical_similarity(first, second)
        qualifies = (
            metrics["length_ratio"] >= 0.50
            and (
                metrics["decision_score"] >= threshold
                or (
                    metrics["token_containment"] >= 0.90
                    and metrics["length_ratio"] >= 0.60
                )
            )
        )
        if not qualifies:
            continue
        qualified_text_pair_count += 1
        evidence = {
            "normalization_version": TEXT_NORMALIZATION_VERSION,
            "blocking_version": BLOCKING_VERSION,
            "threshold": threshold,
            "left_normalized_text": first,
            "right_normalized_text": second,
            **metrics,
        }
        for left_record, right_record in representative_pairs(groups[first], groups[second]):
            collector.add(
                left_record,
                right_record,
                f"{field_name}_near_lexical",
                evidence,
            )
            emitted_record_pair_count += 1

    possible = len(eligible_texts) * (len(eligible_texts) - 1) // 2
    return {
        "eligible_distinct_text_count": len(eligible_texts),
        "possible_all_pairs": possible,
        "blocked_candidate_text_pair_count": len(candidate_text_pairs),
        "qualified_text_pair_count": qualified_text_pair_count,
        "emitted_record_pair_evidence_count": emitted_record_pair_count,
        "minhash_bucket_count": len(buckets),
        "truncated_bucket_count": truncated_bucket_count,
        "skipped_bucket_pair_count": skipped_bucket_pair_count,
        "all_pairs_enumerated": False,
    }


def add_alias_candidates(records: Sequence[AuditRecord], collector: PairCollector) -> Dict[str, Any]:
    groups: Dict[str, List[AuditRecord]] = defaultdict(list)
    for record in records:
        for alias in record.answer_aliases_normalized:
            groups[alias].append(record)
    matched_groups = 0
    for alias, members in sorted(groups.items()):
        unique_members = sorted(
            {item.audit_record_id: item for item in members}.values(), key=lambda item: item.sort_key
        )
        if len(unique_members) < 2 or not any(
            item.role == "current_bundle" for item in unique_members
        ):
            continue
        matched_groups += 1
        evidence = {
            "normalization_version": ANSWER_NORMALIZATION_VERSION,
            "normalized_alias": alias,
            "normalized_alias_sha256": hashlib.sha256(alias.encode("utf-8")).hexdigest(),
        }
        for first, second in spanning_pairs(unique_members):
            collector.add(first, second, "answer_alias_exact", evidence)
    return {
        "normalized_alias_key_count": len(groups),
        "matched_alias_group_count": matched_groups,
    }


class DisjointSet:
    def __init__(self, identifiers: Iterable[str]) -> None:
        self.parent = {identifier: identifier for identifier in identifiers}

    def find(self, identifier: str) -> str:
        parent = self.parent[identifier]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[identifier] = parent
        return parent

    def union(self, first: str, second: str) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        smaller, larger = sorted((root_first, root_second))
        self.parent[larger] = smaller


def compact_member(record: AuditRecord) -> Dict[str, Any]:
    return {
        "audit_record_id": record.audit_record_id,
        "base_fact_id": record.raw.get("base_fact_id"),
        "source_id": record.raw.get("source_id"),
        "candidate_id": record.raw.get("candidate_id"),
        "split_assignment": record.split_assignment,
        "split_group_id": record.split_group_id,
        "answer_raw": record.answer,
        "answer_aliases_raw": list(record.answer_aliases),
        "input_line_number": record.line_number,
        "input_record_sha256": record.record_sha256,
    }


def cross_split_groups(
    grouped: Mapping[str, Sequence[AuditRecord]], max_examples: int
) -> Dict[str, Any]:
    violations: List[Tuple[str, List[AuditRecord]]] = []
    for key, raw_members in grouped.items():
        members = sorted(
            {item.audit_record_id: item for item in raw_members}.values(),
            key=lambda item: item.sort_key,
        )
        splits = {item.split_assignment for item in members if item.split_assignment in ALLOWED_SPLITS}
        if len(splits) > 1:
            violations.append((key, members))
    violations.sort(key=lambda item: item[0])
    affected = {
        member.audit_record_id
        for _, members in violations
        for member in members
    }
    examples = []
    for key, members in violations[:max_examples]:
        examples.append(
            {
                "normalized_group_value": key,
                "normalized_group_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "split_assignments": sorted(
                    {item.split_assignment for item in members if item.split_assignment is not None}
                ),
                "member_count": len(members),
                "members": [compact_member(item) for item in members[:max_examples]],
                "members_truncated": len(members) > max_examples,
            }
        )
    return {
        "cross_split_group_count": len(violations),
        "affected_record_count": len(affected),
        "examples": examples,
        "examples_truncated": len(violations) > max_examples,
    }


def build_split_integrity_audit(
    current: Sequence[AuditRecord],
    max_examples: int,
    candidate_pairs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    primary_answers: Dict[str, List[AuditRecord]] = defaultdict(list)
    alias_keys: Dict[str, List[AuditRecord]] = defaultdict(list)
    split_groups: Dict[str, List[AuditRecord]] = defaultdict(list)
    questions: Dict[str, List[AuditRecord]] = defaultdict(list)
    facts: Dict[str, List[AuditRecord]] = defaultdict(list)
    for record in current:
        if record.answer_normalized:
            primary_answers[record.answer_normalized].append(record)
        for alias in record.answer_aliases_normalized:
            alias_keys[alias].append(record)
        if record.split_group_id:
            split_groups[record.split_group_id].append(record)
        if record.question_normalized:
            questions[record.question_normalized].append(record)
        if record.canonical_fact_normalized:
            facts[record.canonical_fact_normalized].append(record)

    dsu = DisjointSet(record.audit_record_id for record in current)
    by_id = {record.audit_record_id: record for record in current}
    for members in alias_keys.values():
        unique_ids = sorted({item.audit_record_id for item in members})
        for identifier in unique_ids[1:]:
            dsu.union(unique_ids[0], identifier)
    components: Dict[str, List[AuditRecord]] = defaultdict(list)
    for identifier in sorted(by_id):
        components[dsu.find(identifier)].append(by_id[identifier])
    alias_components = {
        root: members for root, members in components.items() if len(members) > 1
    }

    observed_splits = Counter(record.split_assignment or "<missing>" for record in current)
    invalid_records = [
        record for record in current if record.split_assignment not in ALLOWED_SPLITS
    ]
    audits = {
        "normalized_primary_answer": cross_split_groups(primary_answers, max_examples),
        "normalized_answer_alias_key": cross_split_groups(alias_keys, max_examples),
        "answer_alias_connected_component": cross_split_groups(alias_components, max_examples),
        "declared_split_group_id": cross_split_groups(split_groups, max_examples),
        "normalized_exact_question": cross_split_groups(questions, max_examples),
        "normalized_exact_canonical_fact": cross_split_groups(facts, max_examples),
    }
    violation_count = sum(item["cross_split_group_count"] for item in audits.values())
    emitted_cross_split_pairs = [
        pair for pair in candidate_pairs if pair.get("cross_split") is True
    ]
    emitted_cross_split_alias_count = sum(
        "answer_alias_exact" in pair.get("match_types", ())
        for pair in emitted_cross_split_pairs
    )
    emitted_cross_split_text_count = sum(
        any(
            match_type in TEXT_EVIDENCE_TYPES
            for match_type in pair.get("match_types", ())
        )
        for pair in emitted_cross_split_pairs
    )
    candidate_split_leakage_detected = bool(
        violation_count or invalid_records or emitted_cross_split_pairs
    )
    detected_links = bool(violation_count or emitted_cross_split_pairs)
    if invalid_records and detected_links:
        recommended_action = (
            "repair_missing_or_invalid_split_assignments_and_adjudicate_detected_links_"
            "before_split_freeze"
        )
    elif invalid_records:
        recommended_action = "repair_missing_or_invalid_split_assignments_before_split_freeze"
    elif detected_links:
        recommended_action = (
            "adjudicate_detected_links_then_regroup_accepted_components_before_split_freeze"
        )
    else:
        recommended_action = "retain_provisional_status_until_semantic_review"
    return {
        "split_assignments_observed": dict(sorted(observed_splits.items())),
        "missing_or_invalid_split_record_count": len(invalid_records),
        "missing_or_invalid_split_examples": [
            compact_member(item) for item in invalid_records[:max_examples]
        ],
        "checks": audits,
        "cross_split_violation_group_count_sum": violation_count,
        "emitted_cross_split_candidate_pair_count": len(emitted_cross_split_pairs),
        "emitted_cross_split_alias_candidate_pair_count": (
            emitted_cross_split_alias_count
        ),
        "emitted_cross_split_text_candidate_pair_count": emitted_cross_split_text_count,
        "emitted_cross_split_subcounts_are_nonexclusive": True,
        "candidate_split_leakage_detected": candidate_split_leakage_detected,
        "recommended_action": recommended_action,
    }


def count_pairs(candidate_pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    match_types: Counter[str] = Counter()
    scopes: Counter[str] = Counter()
    for pair in candidate_pairs:
        scopes[str(pair["audit_scope"])] += 1
        for match_type in pair["match_types"]:
            match_types[str(match_type)] += 1
    return {
        "candidate_pair_count": len(candidate_pairs),
        "cross_split_candidate_pair_count": sum(
            1 for item in candidate_pairs if item["cross_split"]
        ),
        "match_type_counts": dict(sorted(match_types.items())),
        "scope_counts": dict(sorted(scopes.items())),
    }


def audit(
    input_bundle_path: Path,
    output_dir: Path,
    comparison_canonical_path: Optional[Path] = None,
    question_threshold: float = DEFAULT_QUESTION_THRESHOLD,
    fact_threshold: float = DEFAULT_FACT_THRESHOLD,
    max_bucket_neighbors: int = DEFAULT_MAX_BUCKET_NEIGHBORS,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> Dict[str, Any]:
    if not 0.0 < question_threshold <= 1.0:
        raise ValueError("question_threshold must be in (0, 1]")
    if not 0.0 < fact_threshold <= 1.0:
        raise ValueError("fact_threshold must be in (0, 1]")
    if max_bucket_neighbors < 1:
        raise ValueError("max_bucket_neighbors must be >= 1")
    if max_examples < 1:
        raise ValueError("max_examples must be >= 1")

    input_bundle_path = input_bundle_path.resolve()
    output_dir = output_dir.resolve()
    comparison_canonical_path = (
        comparison_canonical_path.resolve() if comparison_canonical_path else None
    )
    current = load_records(input_bundle_path, "current_bundle")
    comparison = (
        load_records(comparison_canonical_path, "comparison_canonical")
        if comparison_canonical_path
        else []
    )
    all_records = [*current, *comparison]

    collector = PairCollector()
    alias_stats = add_alias_candidates(all_records, collector)
    question_exact = exact_field_candidates(
        all_records, "question", "question_normalized", collector
    )
    fact_exact = exact_field_candidates(
        all_records, "canonical_fact", "canonical_fact_normalized", collector
    )
    question_near = near_field_candidates(
        all_records,
        "question",
        "question_normalized",
        question_threshold,
        max_bucket_neighbors,
        collector,
    )
    fact_near = near_field_candidates(
        all_records,
        "canonical_fact",
        "canonical_fact_normalized",
        fact_threshold,
        max_bucket_neighbors,
        collector,
    )
    candidate_pairs = collector.finalize()

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "lexical_candidate_pairs.jsonl"
    write_jsonl(pairs_path, candidate_pairs)
    pair_counts = count_pairs(candidate_pairs)
    split_integrity = build_split_integrity_audit(
        current,
        max_examples,
        candidate_pairs,
    )
    source_schema_counts = Counter(str(item.raw.get("schema_version")) for item in current)
    summary: Dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "audit_status": "lexical_candidate_generation_complete",
        "semantic_review_status": "not_performed",
        "semantic_review_complete": False,
        "input": {
            "behavior_bundle": {
                "path": str(input_bundle_path),
                "sha256": sha256_file(input_bundle_path),
                "record_count": len(current),
                "schema_version_counts": dict(sorted(source_schema_counts.items())),
            },
            "comparison_canonical": (
                {
                    "path": str(comparison_canonical_path),
                    "sha256": sha256_file(comparison_canonical_path),
                    "record_count": len(comparison),
                }
                if comparison_canonical_path
                else None
            ),
        },
        "policy": {
            "text_normalization_version": TEXT_NORMALIZATION_VERSION,
            "answer_normalization_version": ANSWER_NORMALIZATION_VERSION,
            "blocking_version": BLOCKING_VERSION,
            "question_near_threshold": question_threshold,
            "canonical_fact_near_threshold": fact_threshold,
            "max_bucket_neighbors": max_bucket_neighbors,
            "max_examples": max_examples,
            "minhash_components": MINHASH_COMPONENTS,
            "minhash_band_rows": MINHASH_BAND_ROWS,
            "deterministic": True,
            "network_or_model_used": False,
            "pair_review_contract": {
                "status": "pending_adjudication",
                "answer_alias_exact_requires": [
                    "alias_valid",
                    "same_answer_entity",
                ],
                "alias_valid_scope": (
                    "pair_level_at_least_one_shared_alias_is_valid_for_both_records"
                ),
                "question_or_fact_exact_or_near_requires": [
                    "semantic_duplicate"
                ],
                "mixed_evidence_requires_union": True,
            },
        },
        "candidate_generation": {
            "answer_alias": alias_stats,
            "question_exact": question_exact,
            "question_near": question_near,
            "canonical_fact_exact": fact_exact,
            "canonical_fact_near": fact_near,
            **pair_counts,
            "emitted_representative_pair_count": len(candidate_pairs),
            "exhaustive_record_pair_census": False,
        },
        "split_integrity": split_integrity,
        "artifacts": {
            "candidate_pairs": {
                "path": pairs_path.name,
                "schema_version": PAIR_SCHEMA_VERSION,
                "sha256": sha256_file(pairs_path),
                "record_count": len(candidate_pairs),
            }
        },
        "limitations": {
            "audit_kind": "lexical_candidate_audit_only",
            "semantic_equivalence_decisions": "not_performed",
            "semantic_adjudication_performed": False,
            "semantic_near_duplicate_recall": "not_guaranteed",
            "blocking_is_bounded": True,
            "candidate_pairs_require_review": True,
            "split_freeze_authorized": False,
            "notes": [
                "Lexical candidates can contain false positives and miss paraphrases.",
                "Answer-alias connected components are diagnostics, not adjudicated entity links.",
                (
                    "Candidate-pair counts are emitted representative edges, not an "
                    "exhaustive record-pair census."
                ),
                (
                    "Alias evidence requires alias-validity and same-entity adjudication; "
                    "question/fact lexical evidence requires semantic-duplicate adjudication."
                ),
                "Do not mark semantic_review_complete from this artifact.",
            ],
        },
        "recommended_next_step": split_integrity["recommended_action"],
    }
    summary_path = output_dir / "lexical_audit_summary.json"
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-canonical", type=Path)
    parser.add_argument("--question-threshold", type=float, default=DEFAULT_QUESTION_THRESHOLD)
    parser.add_argument("--fact-threshold", type=float, default=DEFAULT_FACT_THRESHOLD)
    parser.add_argument(
        "--max-bucket-neighbors", type=int, default=DEFAULT_MAX_BUCKET_NEIGHBORS
    )
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit(
        args.input_bundle,
        args.output_dir,
        comparison_canonical_path=args.comparison_canonical,
        question_threshold=args.question_threshold,
        fact_threshold=args.fact_threshold,
        max_bucket_neighbors=args.max_bucket_neighbors,
        max_examples=args.max_examples,
    )
    print(
        json.dumps(
            {
                "summary": str((args.output_dir / "lexical_audit_summary.json").resolve()),
                "candidate_pairs": summary["candidate_generation"]["candidate_pair_count"],
                "candidate_split_leakage_detected": summary["split_integrity"][
                    "candidate_split_leakage_detected"
                ],
                "semantic_review_complete": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
