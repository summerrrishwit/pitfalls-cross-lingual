"""Build a versioned transport-only overlay without changing the frozen original."""
from pathlib import Path


HELPERS = '''
BACKOFF_SCHEDULE = (60, 120, 300)


def transport_failure(detail):
    status = detail.get('http_status')
    return (detail.get('error_code') == 'gateway_zero_usage_error_response'
            or bool(detail.get('gateway_error_hint'))
            or detail.get('error_type') in {'APIConnectionError', 'APITimeoutError', 'InternalServerError', 'RateLimitError'}
            or isinstance(status, int) and (status == 429 or status >= 500))


def verify_overlay(root):
    authority = read(root/'active_worker_runtime.json')
    if authority['parent_manifest_sha256'] != sha(root/'manifest.json'):
        raise ValueError('worker_parent_manifest_changed')
    if Path(authority['entrypoint']).resolve() != Path(__file__).resolve():
        raise ValueError('worker_entrypoint_changed')
    for name, expected in authority['bindings'].items():
        if sha(Path(name)) != expected:
            raise ValueError('worker_runtime_binding_changed')
    return authority


def tail_retry_batches(root, source, manifest):
    claimed = {row['record_id'] for p in (root/'review_packets').glob('packet_*/input.jsonl') for row in rows(p)}
    selected = []
    for index, start in enumerate(range(0, len(source), manifest['batch_size'])):
        path = root/'batches'/f'batch_{index:05d}.json'
        if not path.exists():
            continue
        value = read(path)
        attempts = value.get('attempts', [])
        if (value.get('status') != 'failed' or not attempts or not transport_failure(attempts[-1])
                or value.get('recovery_history') or value.get('transport_retry_epoch')):
            continue
        batch = source[start:start+manifest['batch_size']]
        if value['input_sha256'] != digest(batch) or value.get('translated'):
            raise ValueError('tail_retry_checkpoint_changed')
        if claimed & {row['record_id'] for row in batch}:
            raise ValueError('tail_retry_reviewed_record')
        selected.append((index, batch, path, value))
    # Validate the whole selection before any requeue mutation.
    result = []
    for index, batch, path, value in selected:
        archive = root/'transport_retry_history'/'epoch_1'/path.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise ValueError('tail_retry_archive_already_exists')
        shutil.copyfile(path, archive)
        if sha(archive) != sha(path):
            raise ValueError('tail_retry_archive_copy_failed')
        value['transport_retry_epoch'] = 1
        value['recovery_history'] = [{'recovery_id': 'transport_tail_epoch_1',
                                      'archive': str(archive), 'sha256': sha(archive), 'previous_status': 'failed'}]
        value['status'] = 'normalized' if value.get('normalized') else 'recovery_pending'
        atomic(path, value)
        result.append((index, batch))
    return result

'''


def build(source: str) -> str:
    replacements = [
        ("def work(root: Path) -> None:\n    m=verify(root);", "def work(root: Path) -> None:\n    verify_overlay(root)\n    m=verify(root);"),
        ("health_lock=threading.Lock();failures=0", "health_lock=threading.Lock();failures=0;pause_until=0.;backoff_level=0"),
        ("{'at':now(),'error_code':code,**detail}", "{**detail,'at':now(),'error_code':code}"),
        ("        nonlocal failures", "        nonlocal failures,pause_until,backoff_level"),
        ("        for attempt in range(3):\n            if stop.is_set()", "        for attempt in range(3):\n            while True:\n                with health_lock:remaining=pause_until-time.monotonic()\n                if remaining<=0:break\n                if stop.wait(min(remaining,1)) or (root/'STOP').exists():return None,attempts\n            if stop.is_set()"),
        ("'requested_model':model}\n            exposure", "'requested_model':model,'worker_policy':'transport_continue_v1'}\n            exposure"),
        ("with health_lock:failures=0\n                result", "with health_lock:failures=0;backoff_level=0\n                result"),
        ("                if technical:\n                    with health_lock:failures+=1;current=failures\n                    if current>=16:fatal('systematic_gateway_errors',detail)",
         "                if isinstance(error,RuntimeError) and str(error) not in {'gateway_zero_usage_error_response','response_model_identity_mismatch'}:fatal('unexpected_runtime_error',detail)\n                if technical:\n                    delay=0\n                    with health_lock:\n                        failures+=1\n                        if failures>=16:\n                            delay=BACKOFF_SCHEDULE[min(backoff_level,len(BACKOFF_SCHEDULE)-1)]\n                            backoff_level+=1;failures=0;pause_until=max(pause_until,time.monotonic()+delay)\n                    if delay:exposure({'event':'transport_backoff','seconds':delay,'target_behavior_exposure':False})"),
        ("    with ThreadPoolExecutor(max_workers=m['concurrency']) as executor:\n        futures=[executor.submit(batch_work,i,source[j:j+m['batch_size']]) for i,j in enumerate(range(0,len(source),m['batch_size']))]\n        for future in as_completed(futures):\n            future.result()\n            if stop.is_set() or (root/'STOP').exists():continue\n            with publish_lock:\n                if m['execution_mode']=='pipeline':pack(root,verified_manifest=m)\n                progress(root,'draining_fatal_error' if stop.is_set() else 'running_qwen')",
         "    def run_pass(batches):\n        with ThreadPoolExecutor(max_workers=m['concurrency']) as executor:\n            futures=[executor.submit(batch_work,i,batch) for i,batch in batches]\n            for future in as_completed(futures):\n                future.result()\n                if stop.is_set() or (root/'STOP').exists():continue\n                with publish_lock:\n                    if m['execution_mode']=='pipeline':pack(root,verified_manifest=m)\n                    progress(root,'running_qwen')\n    run_pass([(i,source[j:j+m['batch_size']]) for i,j in enumerate(range(0,len(source),m['batch_size']))])\n    if not stop.is_set() and not (root/'STOP').exists():\n        verify_overlay(root);verify(root)\n        retries=tail_retry_batches(root,source,m)\n        exposure({'event':'transport_tail_retry_started','batch_count':len(retries),'target_behavior_exposure':False})\n        run_pass(retries)"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('unexpected_frozen_source_shape')
        source = source.replace(old, new, 1)
    anchor = '\ndef work(root: Path) -> None:'
    source = source.replace(anchor, '\n'+HELPERS+anchor, 1)
    compile(source, '<resilient-runtime>', 'exec')
    return source


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    if args.target.exists():
        raise ValueError('versioned_runtime_already_exists')
    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(build(args.source.read_text()))
