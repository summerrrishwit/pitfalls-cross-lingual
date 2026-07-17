import os
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.answers import simulate
from utils.tools import logger as utils_logger

logger = utils_logger

def atomic_write_json(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)

def load_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}. Reinitializing.")
        return default

def make_empty_progress(input_file, lang):
    return {
        'input_file': input_file,
        'lang': lang,
        'records': {}
    }

def build_results(progress, filtered_items, models, lang):
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
                    results[language][model] += language_record[model]
                    results['Completed'][language][model] += 1
    return results

def main():
    parser = argparse.ArgumentParser(description="Process language-specific questions for various models.")
    parser.add_argument("--lang", default='Chinese', help="The language to process (e.g., 'French').")
    parser.add_argument("--input_file", default='data/Chinese.json', help="The input JSON file.")
    parser.add_argument("--output_file", default='data/Chinese_results.json', help="The output JSON file.")
    parser.add_argument("--models", type=str, nargs="+", default=['qwen-2.5-7B', 'llama-3.1-8B', 'gemma-2-9B', 'gemma-2-27B', 'llama-3.1-70B', 'qwen-2.5-72B', 'qwen3.7-plus', 'qwen3.7-max', 'gpt-4o-mini', 'gpt-4o', 'yi-lightning', 'o1-mini', 'claude-3.5-sonnet'], help="Models to evaluate. Use: --models qwen3.7-plus qwen3.7-max")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum concurrent eval tasks. Lower this if the API connection is unstable or rate-limited.")
    parser.add_argument("--checkpoint-file", default=None, help="Path to the per-question checkpoint file. Defaults to <output_file>.progress.json.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore an existing checkpoint and start a fresh eval.")
    args = parser.parse_args()

    lang = args.lang
    input_file = args.input_file
    output_file = args.output_file
    checkpoint_file = args.checkpoint_file or f"{output_file}.progress.json"
    models = args.models

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
    if args.no_resume:
        progress = make_empty_progress(input_file, lang)
    else:
        progress = load_json_file(checkpoint_file, make_empty_progress(input_file, lang))
        progress.setdefault('input_file', input_file)
        progress.setdefault('lang', lang)
        progress.setdefault('records', {})

    lock = threading.Lock()

    def save_progress_and_results():
        atomic_write_json(checkpoint_file, progress)
        atomic_write_json(output_file, build_results(progress, filtered_items, models, lang))

    def has_record(entry_idx, language, model):
        return model in progress.get('records', {}).get(str(entry_idx), {}).get(language, {})

    def set_record(entry_idx, language, model, value):
        record = progress.setdefault('records', {}).setdefault(str(entry_idx), {})
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
        score = simulate(language=language if language != '4o' else 'Chinese', question=question, model_list=[model], choices=choices, ground_truth=ground_truth)
        return entry_idx, language, model, score

    tasks = []
    for entry_idx, entry in filtered_items:
        for model in models:
            for language in ['English', lang]:
                if not has_record(entry_idx, language, model):
                    tasks.append((entry_idx, entry, language, model))

    total_slots = len(filtered_items) * len(models) * 2
    completed_slots = total_slots - len(tasks)
    logger.info(f"Resume checkpoint: {checkpoint_file}")
    logger.info(f"Completed slots: {completed_slots}/{total_slots}; remaining: {len(tasks)}")

    if not tasks:
        save_progress_and_results()
        logger.info("All requested eval slots are already complete. Results refreshed from checkpoint.")
    else:
        try:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_tasks = [executor.submit(eval_one, *task) for task in tasks]
                for future in as_completed(future_tasks):
                    entry_idx, language, model, score = future.result()
                    with lock:
                        set_record(entry_idx, language, model, score)
                        save_progress_and_results()
                        completed_slots += 1
                        logger.info(f"Checkpoint saved: {completed_slots}/{total_slots} ({language}, {model}, entry={entry_idx})")
        except KeyboardInterrupt:
            with lock:
                save_progress_and_results()
            logger.warning("Interrupted. Partial progress has been saved; rerun the same command to resume.")
            raise

    final_results = build_results(progress, filtered_items, models, lang)
    print(f"Total Questions = {final_results['Total Questions']}")
    print("Completed")
    print(final_results['Completed'])
    print("English")
    print(final_results['English'])
    print(lang)
    print(final_results[lang])
    logger.info(f"Results have been saved to {output_file}")
    logger.info(f"Checkpoint has been saved to {checkpoint_file}")

if __name__ == "__main__":
    main()
