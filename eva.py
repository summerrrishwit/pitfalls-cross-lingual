import os
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from utils.answers import simulate
from utils.tools import logger as utils_logger

logger = utils_logger

Progress = Dict[str, Any]
FilteredItem = Tuple[int, Dict[str, Any]]

def atomic_write_json(path: str, data: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)

def load_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}. Reinitializing.")
        return default

def make_empty_progress(input_file: str, lang: str) -> Progress:
    return {
        'input_file': input_file,
        'lang': lang,
        'records': {}
    }


def make_empty_model_progress(input_file: str, lang: str, model: str) -> Progress:
    progress = make_empty_progress(input_file, lang)
    progress['model'] = model
    return progress


def safe_filename_component(value: str) -> str:
    return ''.join(
        character if character.isalnum() or character in '._-' else '_'
        for character in value
    )


def get_model_checkpoint_path(checkpoint_dir: str, lang: str, model: str) -> str:
    language_name = safe_filename_component(lang)
    model_name = safe_filename_component(model)
    return os.path.join(
        checkpoint_dir,
        f'{language_name}_results_{model_name}_progress.json',
    )


def extract_model_progress(
    progress: Progress,
    input_file: str,
    lang: str,
    model: str,
) -> Progress:
    model_progress = make_empty_model_progress(input_file, lang, model)
    if progress.get('input_file') not in (None, input_file):
        return model_progress

    for entry_idx, record in progress.get('records', {}).items():
        selected_record = {}
        for language in ('English', lang):
            language_record = record.get(language, {})
            if model in language_record:
                selected_record[language] = {model: language_record[model]}
        if selected_record:
            model_progress['records'][entry_idx] = selected_record
    return model_progress


def merge_model_progresses(
    model_progresses: Dict[str, Progress],
    input_file: str,
    lang: str,
) -> Progress:
    merged_progress = make_empty_progress(input_file, lang)
    for model, model_progress in model_progresses.items():
        for entry_idx, record in model_progress.get('records', {}).items():
            merged_record = merged_progress['records'].setdefault(entry_idx, {})
            for language in ('English', lang):
                language_record = record.get(language, {})
                if model in language_record:
                    merged_record.setdefault(language, {})[model] = language_record[model]
    return merged_progress

def get_score(record: Any) -> float:
    if isinstance(record, dict):
        return float(record.get('score', 0.0))
    return float(record)


def build_results(
    progress: Progress,
    filtered_items: List[FilteredItem],
    models: List[str],
    lang: str,
) -> Dict[str, Any]:
    results = {
        'English': {model: 0 for model in models},
        lang: {model: 0 for model in models},
        'Total Questions': len(filtered_items),
        'Completed': {
            'English': {model: 0 for model in models},
            lang: {model: 0 for model in models},
        }
    }
    for idx, _entry in filtered_items:
        record = progress.get('records', {}).get(str(idx), {})
        for language in ['English', lang]:
            language_record = record.get(language, {})
            for model in models:
                if model in language_record:
                    results[language][model] += get_score(language_record[model])
                    results['Completed'][language][model] += 1
    return results


def compact_language_entry(
    entry: Dict[str, Any],
    language: str,
    lang: str,
) -> Dict[str, Any]:
    if language == 'English':
        return {
            'question': entry.get('question', ''),
            'choices': entry.get('choices', []),
            'ground_truth': entry.get('answer'),
        }
    return {
        'question': entry.get('transquestion', ''),
        'choices': entry.get('transchoices', []),
        'ground_truth': entry.get('transanswer'),
        'language': lang,
    }


def normalize_model_record(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        return record
    score = get_score(record)
    return {
        'status': 'legacy_score_only',
        'raw_response': None,
        'extracted_answer': None,
        'correct': score == 1.0,
        'score': score,
        'extractor_model': None,
    }


def build_detailed_results(
    progress: Progress,
    filtered_items: List[FilteredItem],
    models: List[str],
    lang: str,
) -> Dict[str, Any]:
    items = []
    for idx, entry in filtered_items:
        item = {
            'original_index': idx,
            'English': compact_language_entry(entry, 'English', lang),
            lang: compact_language_entry(entry, lang, lang),
        }
        for metadata_field in ('source', 'category'):
            if metadata_field in entry:
                item[metadata_field] = entry[metadata_field]

        progress_record = progress.get('records', {}).get(str(idx), {})
        for language in ('English', lang):
            language_record = progress_record.get(language, {})
            item[language]['models'] = {
                model: normalize_model_record(language_record[model])
                for model in models
                if model in language_record
            }
        items.append(item)

    return {
        'metadata': {
            'input_file': progress.get('input_file'),
            'target_language': lang,
            'models': models,
            'retained_input_fields': [
                'source', 'category', 'question', 'choices', 'answer',
                'transquestion', 'transchoices', 'transanswer',
            ],
        },
        'summary': build_results(progress, filtered_items, models, lang),
        'items': items,
    }

def main():
    parser = argparse.ArgumentParser(description="Process language-specific questions for various models.")
    parser.add_argument("--lang", default='Chinese', help="The language to process (e.g., 'French').")
    parser.add_argument("--input_file", default='data/Chinese.json', help="The input JSON file.")
    parser.add_argument("--output_file", default='data/Chinese_results.json', help="The output JSON file.")
    parser.add_argument("--detailed-output-file", default=None, help="Per-question model outputs. Defaults to <output stem>.details.json.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Models to evaluate, for example: --models qwen3.5-plus deepseek-v4-pro",
    )
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum concurrent eval tasks. Lower this if the API connection is unstable or rate-limited.")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for per-model checkpoints. Defaults to data/<lang> for outputs under data/.")
    parser.add_argument("--checkpoint-file", default=None, help="Optional legacy combined checkpoint used only as a migration source.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore an existing checkpoint and start a fresh eval.")
    args = parser.parse_args()

    lang = args.lang
    input_file = args.input_file
    output_file = args.output_file
    output_stem, output_extension = os.path.splitext(output_file)
    detailed_output_file = args.detailed_output_file or (
        f"{output_stem}.details{output_extension or '.json'}"
    )
    models = args.models
    checkpoint_dir = args.checkpoint_dir or os.path.join(
        os.path.dirname(output_file) or '.',
        lang,
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_paths = {
        model: get_model_checkpoint_path(checkpoint_dir, lang, model)
        for model in models
    }
    legacy_checkpoint_file = args.checkpoint_file or f"{output_file}.progress.json"

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                logger.error("The input JSON file should be a list.")
                return
    except Exception as e:
        logger.error(f"Failed to load the input file: {e}")
        return

    filtered_items = [(idx, entry) for idx, entry in enumerate(data) if entry['rate_trans'] <= 0.2]
    legacy_progress = make_empty_progress(input_file, lang)
    if not args.no_resume and os.path.exists(legacy_checkpoint_file):
        legacy_progress = load_json_file(legacy_checkpoint_file, legacy_progress)

    model_progresses = {}
    for model in models:
        empty_model_progress = make_empty_model_progress(input_file, lang, model)
        if args.no_resume:
            loaded_progress = empty_model_progress
        elif os.path.exists(checkpoint_paths[model]):
            loaded_progress = load_json_file(checkpoint_paths[model], empty_model_progress)
        else:
            loaded_progress = legacy_progress
        model_progresses[model] = extract_model_progress(
            loaded_progress,
            input_file,
            lang,
            model,
        )

    progress = merge_model_progresses(model_progresses, input_file, lang)

    lock = threading.Lock()

    def save_progress_and_results(updated_model=None, include_details=False):
        models_to_save = [updated_model] if updated_model else models
        for model in models_to_save:
            atomic_write_json(checkpoint_paths[model], model_progresses[model])
        atomic_write_json(output_file, build_results(progress, filtered_items, models, lang))
        if include_details:
            atomic_write_json(
                detailed_output_file,
                build_detailed_results(progress, filtered_items, models, lang),
            )

    def has_record(entry_idx, language, model):
        return model in model_progresses[model].get('records', {}).get(str(entry_idx), {}).get(language, {})

    def set_record(entry_idx, language, model, value):
        for target_progress in (model_progresses[model], progress):
            record = target_progress.setdefault('records', {}).setdefault(str(entry_idx), {})
            record.setdefault(language, {})[model] = value

    def eval_one(entry_idx, entry, language, model):
        if language == 'English':
            question = entry['question']
            choices = entry['choices']
            ground_truth = entry['answer']
        else:
            question = entry.get('transquestion', '')
            choices = entry.get('transchoices', [])
            ground_truth = entry.get('transanswer', '')
        evaluation = simulate(
            language=language if language != '4o' else 'Chinese',
            question=question,
            model_list=[model],
            choices=choices,
            ground_truth=ground_truth,
            return_details=True,
        )
        return entry_idx, language, model, evaluation['models'][model]

    tasks = []
    for entry_idx, entry in filtered_items:
        for model in models:
            for language in ['English', lang]:
                if not has_record(entry_idx, language, model):
                    tasks.append((entry_idx, entry, language, model))

    total_slots = len(filtered_items) * len(models) * 2
    completed_slots = total_slots - len(tasks)
    logger.info(f"Per-model checkpoint directory: {checkpoint_dir}")
    for model in models:
        logger.info(f"Checkpoint for {model}: {checkpoint_paths[model]}")
    logger.info(f"Completed slots: {completed_slots}/{total_slots}; remaining: {len(tasks)}")

    if not tasks:
        save_progress_and_results(include_details=True)
        logger.info("All requested eval slots are already complete. Results refreshed from checkpoint.")
    else:
        try:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_tasks = [executor.submit(eval_one, *task) for task in tasks]
                for future in as_completed(future_tasks):
                    entry_idx, language, model, model_result = future.result()
                    with lock:
                        set_record(entry_idx, language, model, model_result)
                        save_progress_and_results(updated_model=model)
                        completed_slots += 1
                        logger.info(f"Checkpoint saved: {completed_slots}/{total_slots} ({language}, {model}, entry={entry_idx})")
        except KeyboardInterrupt:
            with lock:
                save_progress_and_results(include_details=True)
            logger.warning("Interrupted. Partial progress has been saved; rerun the same command to resume.")
            raise

    save_progress_and_results(include_details=True)
    final_results = build_results(progress, filtered_items, models, lang)
    print(f"Total Questions = {final_results['Total Questions']}")
    print("Completed")
    print(final_results['Completed'])
    print("English")
    print(final_results['English'])
    print(lang)
    print(final_results[lang])
    logger.info(f"Results have been saved to {output_file}")
    logger.info(f"Detailed results have been saved to {detailed_output_file}")
    logger.info(f"Per-model checkpoints have been saved to {checkpoint_dir}")

if __name__ == "__main__":
    main()
