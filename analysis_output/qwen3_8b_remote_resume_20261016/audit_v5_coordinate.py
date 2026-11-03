"""Remote preflight: compare actual layer inputs, returned states and score paths."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'scripts'))
from qwen3_open_repair_common import Engine, suffix, write_json
from run_qwen3_prefill_repair import PrefillIntervention

root = Path(__file__).resolve().parent
engine = Engine(root/'repair_config_v5.json', 'coordinate-audit')
row = next(r for r in engine.inputs if r['option_free_id'] == '359b2bd65150301dc297b5b7')
tokens = row['prompt_token_ids']
captures = {}
handles = []
for layer in range(36):
    def hook(module, args, layer_id=layer):
        captures[layer_id] = args[0][0, len(tokens)-1].detach().clone().cpu()
    handles.append(engine.model.model.layers[layer].register_forward_pre_hook(hook))
with engine.torch.inference_mode():
    output = engine.model.model(input_ids=engine.torch.tensor([tokens], device='cuda'),
                                use_cache=False, output_hidden_states=True)
returned = engine.torch.stack([h[0, -1].cpu() for h in output.hidden_states[:-1]])
prefill = dict(captures)
gold = next(g for g in row['answer_groups'] if g['option_id'] == 'gold')['aliases_zh'][0]
tail = suffix(engine.tokenizer, row['rendered_prompt'], tokens, gold)
with engine.torch.inference_mode():
    engine.model(input_ids=engine.torch.tensor([tokens+tail], device='cuda'), use_cache=False)
teacher = dict(captures)
for handle in handles:
    handle.remove()
cached = None
for path in sorted((root/'repair-v3/open-trajectories').glob('chunk-*.pt')):
    data = engine.torch.load(path, map_location='cpu', weights_only=True)
    if row['option_free_id'] in data['ids']:
        cached = data['states'][data['ids'].index(row['option_free_id'])]
        break
assert cached is not None
checks = []
for layer in (12, 20, 28, 32):
    a = prefill[layer].float()
    checks.append({'layer': layer,
        'returned_vs_actual_max_abs': float((returned[layer].float()-a).abs().max()),
        'cached_vs_actual_max_abs': float((cached[layer].float()-a).abs().max()),
        'teacher_vs_prefill_max_abs': float((teacher[layer].float()-a).abs().max()),
        'teacher_vs_prefill_relative_l2': float((teacher[layer].float()-a).norm()/a.norm())})
baseline = engine.sequence_scores(row, {})['gold_margin']
zero = engine.torch.zeros(4096)
with PrefillIntervention(engine.model, {12: zero}, len(tokens)):
    zero_margin = engine.sequence_scores(row, {})['gold_margin']
report = {'input_id': row['option_free_id'], 'coordinate_checks': checks,
          'fresh_noop_vs_saved_margin_delta': abs(baseline-engine.baseline[row['option_free_id']]['gold_minus_designated_distractor_margin']),
          'additive_zero_vs_noop_margin_delta': abs(zero_margin-baseline)}
write_json(root/'repair-v5/coordinate_audit.json', report)
print(report, flush=True)
