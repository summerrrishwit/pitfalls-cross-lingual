"""Derive an error-observability-only runtime from an immutable SSE parent."""
from pathlib import Path


def build_worker(source: str) -> str:
    changes = [
        ('    from qwen_stream_transport import stream_completion\n',
         '    from qwen_stream_transport import stream_completion\n    from sanitized_api_error import api_error_diagnostics\n'),
        ("                detail.update(status='failed',**safe_error(error),latency_seconds=round(time.monotonic()-start,3))\n",
         "                detail.update(status='failed',**safe_error(error),latency_seconds=round(time.monotonic()-start,3))\n                detail.update(api_error_diagnostics(error,key,base))\n"),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError('unexpected_parent_source_shape')
        source = source.replace(old, new, 1)
    compile(source, '<diagnostic-runtime>', 'exec')
    return source


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parent', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    args.target.mkdir(parents=True, exist_ok=False)
    for path in args.parent.glob('*.py'):
        (args.target/path.name).write_bytes(path.read_bytes())
    (args.target/'run_zh_candidate_pipeline.py').write_text(build_worker((args.parent/'run_zh_candidate_pipeline.py').read_text()))
    for name in ['sanitized_api_error.py', 'build_zh_error_diagnostics_runtime.py']:
        (args.target/name).write_bytes(Path(__file__).with_name(name).read_bytes())
