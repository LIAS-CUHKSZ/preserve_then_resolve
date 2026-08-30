from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from dino_m2m.cli import _apply_model_path_fallbacks, _build_parser, main


class CliTests(unittest.TestCase):
    def test_debias_rank_is_the_public_cli_term(self) -> None:
        parser = _build_parser({})
        args = parser.parse_args(["match", "--debias-ranks", "100", "200"])
        self.assertEqual(args.svd_components, [100, 200])
        match_parser = next(
            action for action in parser._actions if action.dest == "command"
        ).choices["match"]
        help_text = match_parser.format_help()
        self.assertIn("--debias-ranks RANK", help_text)
        self.assertNotIn("svd-components", help_text)
        configured = _build_parser(
            {"matching": {"debias_ranks": [300]}}
        ).parse_args(["match"])
        self.assertEqual(configured.svd_components, [300])

    def test_model_agnostic_source_and_weight_aliases(self) -> None:
        parser = _build_parser({})
        args = parser.parse_args(
            [
                "extract-dino",
                "--dinov2-source",
                "/tmp/dinov2",
                "--dinov2-weights",
                "/tmp/dinov2.pth",
            ]
        )
        self.assertEqual(str(args.source), "/tmp/dinov2")
        self.assertEqual(str(args.weights), "/tmp/dinov2.pth")

    def test_cli_model_override_selects_matching_family_config_paths(self) -> None:
        config = {
            "paths": {
                "dinov2_source": Path("/tmp/config-dinov2"),
                "dinov2_weights": Path("/tmp/config-dinov2.pth"),
                "dinov3_source": Path("/tmp/config-dinov3"),
                "dinov3_weights": Path("/tmp/config-dinov3.pth"),
            }
        }
        args = _build_parser(config).parse_args(
            ["extract-dino", "--model", "dinov2_vitl14_reg"]
        )
        _apply_model_path_fallbacks(args, config)
        self.assertEqual(args.source, config["paths"]["dinov2_source"])
        self.assertEqual(args.weights, config["paths"]["dinov2_weights"])

    def test_root_help_loads_without_vision_dependencies(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("extract-dino", output.getvalue())
        self.assertIn("fit-bias-for-pairs", output.getvalue())
        self.assertIn("eval-dense-correspondence", output.getvalue())
        self.assertIn("summarize-dense-correspondence", output.getvalue())
        self.assertIn("eval-estimation", output.getvalue())


if __name__ == "__main__":
    unittest.main()
