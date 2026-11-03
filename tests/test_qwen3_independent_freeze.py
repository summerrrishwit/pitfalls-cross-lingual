import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_qwen3_open_independent import freeze


class FreezeAuthorizationTests(unittest.TestCase):
    def test_failed_development_rule_cannot_unlock_independent_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=root/"config.json";config.write_text(json.dumps({"output_dir":directory}))
            for status in ("G2_failed_random_comparison", "G2_failed_no_eligible_generation_candidate",
                           "G2_not_confirmable_missing_primary_judgments"):
                with self.subTest(status=status):
                    (root/"selected_rule.json").write_text(json.dumps({"status":status}))
                    with self.assertRaisesRegex(ValueError,"G2 selection did not pass"):
                        freeze(config)
                    self.assertFalse((root/"repair_rule_frozen.json").exists())


if __name__=="__main__":unittest.main()
