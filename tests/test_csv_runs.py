"""Run with: .nav/bin/python -m unittest discover -s tests"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import csv_runs


class CsvRunIdentityTest(unittest.TestCase):
    def test_visualization_uses_the_registered_model_tag(self):
        with tempfile.TemporaryDirectory() as scratch:
            runs = Path(scratch) / "model_runs"
            with patch.object(csv_runs, "RUNS", runs), patch.object(csv_runs, "ACTIVE", runs / "active.json"):
                tag = "HardPos_embed16_context10_epochs1000_"
                names = (
                    f"cosine_similarity_matrix_gpt_pre_train_{tag}.csv",
                    f"cosine_similarity_matrix_gpt_{tag}_post_train_.csv",
                    f"{tag}_train_metrics.csv",
                )
                for name in names:
                    csv_runs.save_csv_path(tag, name).write_text("value\n1\n")
                csv_runs.register_ps_mapping(tag, {"PS_A": 0, "PS_B": 1})
                self.assertEqual(csv_runs.load_ps_mapping(tag), {"PS_A": 0, "PS_B": 1})
                with self.assertRaises(ValueError):
                    csv_runs.register_ps_mapping(tag, {"PS_A": 1, "PS_B": 0})
                csv_runs.register_active_run("fixed", tag)
                self.assertEqual(csv_runs.active_model_tag("fixed"), tag)
                self.assertEqual(csv_runs.load_csv_path(tag, names[0]).read_text(), "value\n1\n")

                # Re-running the same tagged model may change numerical values.
                csv_runs.save_csv_path(tag, names[0]).write_text("value\n2\n")
                self.assertEqual(csv_runs.load_csv_path(tag, names[0]).read_text(), "value\n2\n")

                with self.assertRaises(ValueError):
                    csv_runs.load_csv_path(tag, "cosine_similarity_matrix_gpt_pre_train_HardPos_.csv")
                with self.assertRaises(ValueError):
                    csv_runs.save_csv_path("HardPos_", f"cosine_similarity_matrix_gpt_pre_train_{tag}.csv")
                with self.assertRaises(FileNotFoundError):
                    csv_runs.load_csv_path("HardPos_", "cosine_similarity_matrix_gpt_pre_train_HardPos_.csv")
                snapshot = f"full_fwd_train_epoch_199_{tag}"
                csv_runs.save_full_fwd_path(tag, snapshot).write_bytes(b"snapshot")
                self.assertEqual(csv_runs.load_full_fwd_path(tag, snapshot).read_bytes(), b"snapshot")
                with self.assertRaises(ValueError):
                    csv_runs.load_full_fwd_path(tag, "full_fwd_train_epoch_199_HardPos_")
                with self.assertRaises(ValueError):
                    csv_runs.save_full_fwd_path("HardPos_", snapshot)
                with self.assertRaises(FileNotFoundError):
                    csv_runs.load_full_fwd_path(tag, f"full_fwd_{tag}")

                checkpoint = f"gpt_model_{tag}.pt"
                csv_runs.save_checkpoint_path(tag, checkpoint).write_bytes(b"weights")
                self.assertEqual(csv_runs.load_checkpoint_path(tag, checkpoint).read_bytes(), b"weights")
                with self.assertRaises(ValueError):
                    csv_runs.load_checkpoint_path(tag, "gpt_model_HardPos_.pt")

                generation = f"{tag}_gpt_generation_tokens_507.pkl"
                csv_runs.save_generation_path(tag, generation).write_bytes(b"tokens")
                self.assertEqual(csv_runs.load_generation_path(tag, generation).read_bytes(), b"tokens")
                with self.assertRaises(ValueError):
                    csv_runs.load_generation_path(tag, "HardPos__gpt_generation_tokens_507.pkl")

                analysis = f"{tag}single_token_hidden_activations.parquet"
                csv_runs.save_analysis_path(tag, analysis).write_bytes(b"activations")
                self.assertEqual(csv_runs.load_analysis_path(tag, analysis).read_bytes(), b"activations")
                with self.assertRaises(ValueError):
                    csv_runs.load_analysis_path(tag, "HardPos_single_token_hidden_activations.parquet")


if __name__ == "__main__":
    unittest.main()
