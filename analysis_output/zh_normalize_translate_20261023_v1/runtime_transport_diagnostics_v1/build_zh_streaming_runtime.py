"""Derive streaming transport from the previously frozen continuation overlay."""
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError('unexpected_parent_source_shape')
    return source.replace(old, new, 1)


def build_worker(source: str) -> str:
    source = replace_once(source, '    from openai import OpenAI\n',
                          '    from openai import OpenAI\n    from qwen_stream_transport import stream_completion\n')
    source = replace_once(source, "response=c.chat.completions.create(model=model,", 
                          "response=stream_completion(c,expected_model=m['expected_response_model'],model=model,")
    source = replace_once(source, "                content=response.choices[0].message.content or ''\n",
                          "                detail.update(response.streaming_diagnostics)\n                content=response.choices[0].message.content or ''\n")
    source = replace_once(source, "'worker_policy':'transport_continue_v1'", "'worker_policy':'transport_stream_v1'")
    source = replace_once(source, "                status=getattr(error,'status_code',None)\n",
                          "                if isinstance(error,RuntimeError) and str(error)=='response_model_identity_mismatch':fatal('response_model_identity_mismatch',detail)\n                status=getattr(error,'status_code',None)\n")
    compile(source, '<streaming-worker>', 'exec')
    return source


def build_probe(source: str) -> str:
    source = replace_once(source, '    from openai import OpenAI\n',
                          '    from openai import OpenAI\n    from qwen_stream_transport import stream_completion\n')
    source = replace_once(source, 'response = client.chat.completions.create(\n',
                          "response = stream_completion(client, expected_model='Qwen-3.8-Max',\n")
    # The probe needs the complete message and the finish marker, just like before.
    source = replace_once(source, '"finish_reason": response.choices[0].finish_reason,',
                          '"finish_reason": response.streaming_diagnostics["finish_reason"],\n                    **response.streaming_diagnostics,')
    source = replace_once(source, '"probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),',
                          '"probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),\n        "transport_helper_sha256": hashlib.sha256(Path(__file__).with_name("qwen_stream_transport.py").read_bytes()).hexdigest(),')
    compile(source, '<streaming-probe>', 'exec')
    return source


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('parent', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    args.target.mkdir(parents=True, exist_ok=False)
    (args.target/'run_zh_candidate_pipeline.py').write_text(build_worker((args.parent/'run_zh_candidate_pipeline.py').read_text()))
    (args.target/'probe_relation_expansion_api.py').write_text(build_probe((args.parent/'probe_relation_expansion_api.py').read_text()))
    for name in ['qwen_stream_transport.py','build_zh_streaming_runtime.py']:
        (args.target/name).write_bytes(Path(__file__).with_name(name).read_bytes())
