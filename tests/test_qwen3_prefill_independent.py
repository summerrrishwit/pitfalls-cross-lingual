import sys
from pathlib import Path
import json
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_qwen3_prefill_independent import freeze, independent


class PrefillIndependentGateTests(unittest.TestCase):
    def test_no_independent_rule_without_G2(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            cfg = out/"config.json"
            cfg.write_text(json.dumps({"output_dir": temporary}))
            (out/"selected_rule.json").write_text(json.dumps({"status": "G2_failed_no_eligible_generation_candidate"}))
            with self.assertRaisesRegex(ValueError, "G2 selection did not pass"):
                freeze(cfg)
            self.assertFalse((out/"repair_rule_frozen.json").exists())

    def test_old_every_token_rule_cannot_enter_prefill_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            cfg = out/"config.json"
            cfg.write_text(json.dumps({"output_dir": temporary, "remote_root": temporary}))
            (out/"repair_rule_frozen.json").write_text(json.dumps({"status": "g2_repair_rule_frozen", "hook_scope": "every_token"}))
            with self.assertRaisesRegex(ValueError, "prefill G2 rule not frozen"):
                independent(cfg, "validation")
            self.assertFalse((out/"validation").exists())


if __name__ == "__main__":
    unittest.main()
