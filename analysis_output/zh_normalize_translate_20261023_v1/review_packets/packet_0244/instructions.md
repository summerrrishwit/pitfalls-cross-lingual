Review EN->ZH translation ONLY, not factual truth or relation admission. Read the complete
English and Chinese fields of EVERY packet row. Review in bounded groups (at most5 rows per tool output),
without truncation. No outside lookup. Check all fields, entity/answer/alias identity and order, qualifiers,
meaning, usable natural Chinese and no answer leaked in prompt. Minor stylistic preferences alone do not
justify rejection. Copy record_id,input_sha256,english_sha256,translation_sha256 exactly.
Return one JSONL row per packet item: copied hashes, verdict(accept|reject|defer), checks(object with
translation_equivalent,natural,answer_not_leaked,identity_preserved booleans), reason(nonempty),issues(list
of strings),human_gold:false. accept iff all4 checks true. Use defer for genuinely uncertain translation
cases, reject for clear material errors. Do NOT revise the English, certify its relation/fact, or add facts.
Luna reviews every translated candidate. Only translation uncertainty is routed to a different Sol agent;
Sol must also copy review_sha256 and may accept/reject/defer. Sol is model adjudication, not human gold.
