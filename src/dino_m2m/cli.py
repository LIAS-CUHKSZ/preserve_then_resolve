"""Command-line entry point for extraction and matching."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import apply_dataset_override, get_value, load_configs


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="YAML configuration file; repeat to layer paper and local path profiles.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Named dataset override from the configuration (for example METU-CC).",
    )
    return parser


def _path_default(config: dict[str, Any], key: str) -> Path | None:
    value = get_value(config, key)
    return Path(value) if value is not None else None


def _model_path_default(config: dict[str, Any], generic_key: str) -> Path | None:
    """Return a model-agnostic path; family fallback is deferred until parsing."""
    return _path_default(config, f"paths.{generic_key}")


def _apply_model_path_fallbacks(
    args: argparse.Namespace, config: dict[str, Any]
) -> None:
    """Resolve legacy family paths after command-line ``--model`` is known."""
    if not hasattr(args, "model"):
        return
    model_name = str(args.model)
    family = "dinov2" if model_name.startswith("dinov2_") else "dinov3"
    if hasattr(args, "source") and args.source is None:
        args.source = _path_default(config, f"paths.{family}_source")
    if hasattr(args, "weights") and args.weights is None:
        args.weights = _path_default(config, f"paths.{family}_weights")


def _build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="dino-m2m",
        description="DINO progressive many-to-many matching tools",
        parents=[common],
    )
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract-dino", help="Extract selected DINO layers")
    extract.add_argument("--input-root", type=Path, default=_path_default(config, "paths.image_root"))
    extract.add_argument("--pair-file", type=Path, default=_path_default(config, "paths.pair_file"))
    extract.add_argument("--image-list", type=Path)
    extract.add_argument("--output-root", type=Path, default=_path_default(config, "paths.dino_root"))
    extract.add_argument(
        "--source",
        "--dinov3-source",
        "--dinov2-source",
        dest="source",
        type=Path,
        default=_model_path_default(config, "dino_source"),
    )
    extract.add_argument(
        "--weights",
        "--dinov3-weights",
        "--dinov2-weights",
        dest="weights",
        type=Path,
        default=_model_path_default(config, "dino_weights"),
    )
    extract.add_argument("--model", default=get_value(config, "model.name", "dinov3_vitl16"))
    extract.add_argument(
        "--layer", type=int, nargs="+", default=[get_value(config, "model.layer", 19)]
    )
    extract.add_argument(
        "--batch-size", type=int, default=get_value(config, "extraction.batch_size", 4)
    )
    extract.add_argument(
        "--long-edge", type=int, default=get_value(config, "extraction.long_edge", 1024)
    )
    extract.add_argument(
        "--downscale-only",
        action=argparse.BooleanOptionalAction,
        default=get_value(config, "resize.downscale_only", False),
    )
    extract.add_argument("--device", default="auto")
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=get_value(config, "extraction.compile_model", False),
    )

    bias = commands.add_parser("fit-bias", help="Estimate positional-bias bases")
    bias.add_argument(
        "--source",
        "--dinov3-source",
        "--dinov2-source",
        dest="source",
        type=Path,
        default=_model_path_default(config, "dino_source"),
    )
    bias.add_argument(
        "--weights",
        "--dinov3-weights",
        "--dinov2-weights",
        dest="weights",
        type=Path,
        default=_model_path_default(config, "dino_weights"),
    )
    bias.add_argument("--output", type=Path)
    bias.add_argument("--model", default=get_value(config, "model.name", "dinov3_vitl16"))
    bias.add_argument("--layer", type=int, nargs="+", default=[get_value(config, "model.layer", 19)])
    bias.add_argument("--image-height", type=int, required=True)
    bias.add_argument("--image-width", type=int, required=True)
    bias.add_argument(
        "--debias-ranks",
        dest="svd_components",
        type=int,
        nargs="+",
        metavar="RANK",
        default=get_value(config, "matching.debias_ranks", [500]),
        help="Positional-debias ranks to fit.",
    )
    bias.add_argument("--device", default="auto")
    bias.add_argument("--save-json", action="store_true")

    batch_bias = commands.add_parser(
        "fit-bias-for-pairs",
        help="Fit every positional basis required by dimension-annotated pair CSVs",
    )
    batch_bias.add_argument("--pairs-root", type=Path, required=True)
    batch_bias.add_argument("--pair-pattern", default="pairs_*.csv")
    batch_bias.add_argument(
        "--source",
        "--dinov3-source",
        "--dinov2-source",
        dest="source",
        type=Path,
        default=_model_path_default(config, "dino_source"),
    )
    batch_bias.add_argument(
        "--weights",
        "--dinov3-weights",
        "--dinov2-weights",
        dest="weights",
        type=Path,
        default=_model_path_default(config, "dino_weights"),
    )
    batch_bias.add_argument(
        "--output-root", type=Path, default=_path_default(config, "paths.basis_root")
    )
    batch_bias.add_argument("--model", default=get_value(config, "model.name", "dinov3_vitl16"))
    batch_bias.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=[get_value(config, "model.layer", 19)],
    )
    batch_bias.add_argument(
        "--long-edge", type=int, default=get_value(config, "extraction.long_edge", 1024)
    )
    batch_bias.add_argument(
        "--downscale-only",
        action=argparse.BooleanOptionalAction,
        default=get_value(config, "resize.downscale_only", False),
    )
    batch_bias.add_argument(
        "--debias-ranks",
        dest="svd_components",
        type=int,
        nargs="+",
        metavar="RANK",
        default=get_value(config, "matching.debias_ranks", [500]),
        help="Positional-debias ranks to fit.",
    )
    batch_bias.add_argument(
        "--basis-filename-template",
        default=get_value(
            config,
            "matching.basis_filename_template",
            "dinov3_vitl16_{height}x{width}_basis.pt",
        ),
    )
    batch_bias.add_argument("--device", default="auto")
    batch_bias.add_argument("--existing", choices=("error", "skip", "overwrite"), default="error")
    batch_bias.add_argument("--save-json", action="store_true")

    keypoints = commands.add_parser(
        "extract-keypoints", help="Precompute validated SuperPoint caches"
    )
    _add_pair_and_superpoint_args(keypoints, config)
    keypoints.add_argument("--include-descriptors", action="store_true")
    keypoints.add_argument("--overwrite", action="store_true")
    keypoints.add_argument("--max-pairs", type=int)

    match = commands.add_parser("match", help="Run interpolation, debiasing, and association")
    _add_pair_and_superpoint_args(match, config)
    match.add_argument("--dino-root", type=Path, default=_path_default(config, "paths.dino_root"))
    match.add_argument(
        "--weights",
        "--dino-weights",
        "--dinov3-weights",
        "--dinov2-weights",
        dest="weights",
        type=Path,
        default=_model_path_default(config, "dino_weights"),
    )
    match.add_argument(
        "--source",
        "--dinov3-source",
        "--dinov2-source",
        dest="source",
        type=Path,
        default=_model_path_default(config, "dino_source"),
        help="Optional local model checkout used to record source revision provenance.",
    )
    match.add_argument("--model", default=get_value(config, "model.name", "dinov3_vitl16"))
    match.add_argument("--layer", type=int, default=get_value(config, "model.layer", 19))
    match.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Optional assertion; otherwise derived from the backbone profile.",
    )
    match.add_argument(
        "--correction",
        choices=("auto", "none", "positional-debias"),
        default=get_value(config, "matching.correction", "auto"),
    )
    match.add_argument("--basis-root", type=Path, default=_path_default(config, "paths.basis_root"))
    match.add_argument(
        "--output-root", type=Path, default=_path_default(config, "paths.association_root")
    )
    match.add_argument(
        "--basis-filename-template",
        default=get_value(
            config,
            "matching.basis_filename_template",
            "dinov3_vitl16_{height}x{width}_basis.pt",
        ),
    )
    match.add_argument(
        "--debias-ranks",
        dest="svd_components",
        type=int,
        nargs="+",
        metavar="RANK",
        default=get_value(config, "matching.debias_ranks", [500]),
        help="Positional-debias ranks to evaluate.",
    )
    match.add_argument(
        "--max-k", type=int, nargs="+", default=get_value(config, "matching.max_k", [5])
    )
    match.add_argument(
        "--association-upperbound",
        type=int,
        default=get_value(config, "matching.association_upperbound", 2048),
    )
    match.add_argument("--max-pairs", type=int)
    match.add_argument("--existing", choices=("overwrite", "skip"), default="overwrite")
    match.add_argument(
        "--compute-missing-keypoints",
        action="store_true",
        help="Use an externally installed LightGlue SuperPoint for cache misses.",
    )
    match.add_argument("--keypoint-cache-overwrite", action="store_true")

    for name, module, help_text in (
        (
            "eval-dense-correspondence",
            "evaluation.dense_correspondence.evaluate",
            "Generate dense directional/mutual GT-rank CDF shards",
        ),
        (
            "summarize-dense-correspondence",
            "evaluation.dense_correspondence.summarize",
            "Aggregate and plot dense GT-rank CDF shards",
        ),
        ("eval-estimation", "evaluation.estimation.cli", "Run six-dataset pose evaluation"),
    ):
        evaluation = commands.add_parser(name, help=help_text)
        evaluation.set_defaults(evaluation_module=module)
        evaluation.add_argument("evaluation_args", nargs=argparse.REMAINDER)
    return parser


def _add_pair_and_superpoint_args(parser: argparse.ArgumentParser, config: dict[str, Any]) -> None:
    parser.add_argument("--pair-file", type=Path, default=_path_default(config, "paths.pair_file"))
    parser.add_argument("--image-root", type=Path, default=_path_default(config, "paths.image_root"))
    parser.add_argument(
        "--keypoint-cache-root",
        type=Path,
        default=_path_default(config, "paths.keypoint_cache_root"),
    )
    parser.add_argument(
        "--max-num-keypoints",
        type=int,
        default=get_value(config, "superpoint.max_num_keypoints", 2048),
    )
    parser.add_argument(
        "--long-edge", type=int, default=get_value(config, "superpoint.long_edge", 1024)
    )
    parser.add_argument(
        "--downscale-only",
        action=argparse.BooleanOptionalAction,
        default=get_value(config, "resize.downscale_only", False),
    )
    parser.add_argument(
        "--superpoint-weights",
        type=Path,
        default=_path_default(config, "paths.superpoint_weights"),
    )
    parser.add_argument(
        "--superpoint-weights-id",
        default=get_value(config, "superpoint.weights_id"),
        help=(
            "Expected identity recorded in a cache produced with LightGlue's default "
            "weights; explicit checkpoint files are identified by SHA-256 automatically."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-legacy-keypoint-cache",
        action="store_true",
        help=(
            "Use old cache filenames/metadata after manually verifying that their "
            "resize and weight settings match this run."
        ),
    )


def _require_paths(parser: argparse.ArgumentParser, args: argparse.Namespace, names: Sequence[str]) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if getattr(args, name) is None]
    if missing:
        parser.error("Missing required paths (pass them directly or through --config): " + ", ".join(missing))


def _superpoint_config(args: argparse.Namespace):
    from .superpoint import SuperPointConfig

    return SuperPointConfig(
        max_num_keypoints=args.max_num_keypoints,
        long_edge=args.long_edge,
        downscale_only=args.downscale_only,
        weights=args.superpoint_weights,
        expected_weights_id=args.superpoint_weights_id,
    )


def _extract_keypoints(args: argparse.Namespace) -> int:
    from .matching import resolve_device
    from .pairs import read_pairs, unique_images
    from .superpoint import CacheBackedSuperPoint, ExternalLightGlueSuperPoint

    config = _superpoint_config(args)
    device = resolve_device(args.device)
    adapter = ExternalLightGlueSuperPoint(device, config)
    cache = CacheBackedSuperPoint(
        args.image_root,
        args.keypoint_cache_root,
        config,
        adapter,
        overwrite=args.overwrite,
        allow_legacy_cache=args.allow_legacy_keypoint_cache,
    )
    images = unique_images(read_pairs(args.pair_file, args.max_pairs))
    for image_rel in images:
        cache.load_or_extract(image_rel, include_descriptors=args.include_descriptors)
    print(
        f"Prepared {len(images)} SuperPoint caches under {args.keypoint_cache_root}; "
        f"weights_id={cache.weights_id}"
    )
    return 0


def _dispatch_evaluation(module_name: str, arguments: list[str]) -> int:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Evaluation frontend {module_name!r} is not installed in this checkout."
        ) from exc
    entry = getattr(module, "main", None)
    if entry is None:
        raise RuntimeError(f"Evaluation frontend {module_name!r} has no main()")
    result = entry(arguments)
    return int(result) if result is not None else 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap, _ = _common_parser().parse_known_args(argv)
    config = load_configs(bootstrap.config) if bootstrap.config else {}
    config = apply_dataset_override(config, bootstrap.dataset)
    parser = _build_parser(config)
    args = parser.parse_args(argv)
    _apply_model_path_fallbacks(args, config)

    if args.command == "extract-dino":
        _require_paths(parser, args, ("input_root", "output_root", "source", "weights"))
        from .dino import ExtractionOptions, extract_dino

        written = extract_dino(
            ExtractionOptions(
                input_root=args.input_root,
                output_root=args.output_root,
                source=args.source,
                weights=args.weights,
                layers=tuple(args.layer),
                model_name=args.model,
                pair_file=args.pair_file,
                image_list=args.image_list,
                batch_size=args.batch_size,
                long_edge=args.long_edge,
                downscale_only=args.downscale_only,
                device=args.device,
                overwrite=args.overwrite,
                compile_model=args.compile_model,
            )
        )
        print(f"Wrote {written} DINO descriptor files under {args.output_root}")
        return 0

    if args.command == "fit-bias":
        _require_paths(parser, args, ("source", "weights", "output"))
        from .dino import fit_bias

        outputs = fit_bias(
            source=args.source,
            weights=args.weights,
            output=args.output,
            model_name=args.model,
            layers=tuple(args.layer),
            image_height=args.image_height,
            image_width=args.image_width,
            components=tuple(sorted(set(args.svd_components))),
            device_arg=args.device,
            save_json=args.save_json,
        )
        print("Wrote positional bases: " + ", ".join(str(path) for path in outputs))
        return 0

    if args.command == "fit-bias-for-pairs":
        _require_paths(parser, args, ("pairs_root", "source", "weights", "output_root"))
        from .dino import fit_bias_for_pair_files

        if args.pairs_root.is_file():
            pair_files = (args.pairs_root,)
        elif args.pairs_root.is_dir():
            pair_files = tuple(sorted(args.pairs_root.glob(f"**/{args.pair_pattern}")))
        else:
            parser.error(f"pairs root does not exist: {args.pairs_root}")
        if not pair_files:
            parser.error(
                f"no {args.pair_pattern!r} files found under {args.pairs_root}"
            )
        outputs = fit_bias_for_pair_files(
            pair_files=pair_files,
            source=args.source,
            weights=args.weights,
            output_root=args.output_root,
            filename_template=args.basis_filename_template,
            model_name=args.model,
            layers=tuple(args.layer),
            long_edge=args.long_edge,
            downscale_only=args.downscale_only,
            components=tuple(args.svd_components),
            device_arg=args.device,
            existing=args.existing,
            save_json=args.save_json,
        )
        if outputs:
            print("Wrote positional bases: " + ", ".join(str(path) for path in outputs))
        else:
            print("All required positional bases already exist; wrote 0 files")
        return 0

    if args.command == "extract-keypoints":
        _require_paths(parser, args, ("pair_file", "image_root", "keypoint_cache_root"))
        return _extract_keypoints(args)

    if args.command == "match":
        _require_paths(
            parser,
            args,
            (
                "pair_file",
                "image_root",
                "dino_root",
                "weights",
                "keypoint_cache_root",
                "output_root",
            ),
        )
        from .pipeline import MatchOptions, run_matching

        summary = run_matching(
            MatchOptions(
                pair_file=args.pair_file,
                image_root=args.image_root,
                dino_root=args.dino_root,
                keypoint_cache_root=args.keypoint_cache_root,
                output_root=args.output_root,
                weights=args.weights,
                source=args.source,
                model_name=args.model,
                layer=args.layer,
                patch_size=args.patch_size,
                correction=args.correction,
                basis_root=args.basis_root,
                basis_filename_template=args.basis_filename_template,
                svd_components=tuple(args.svd_components),
                max_ks=tuple(args.max_k),
                association_upperbound=args.association_upperbound,
                device=args.device,
                max_pairs=args.max_pairs,
                existing=args.existing,
                compute_missing_keypoints=args.compute_missing_keypoints,
                keypoint_cache_overwrite=args.keypoint_cache_overwrite,
                allow_legacy_keypoint_cache=args.allow_legacy_keypoint_cache,
                superpoint=_superpoint_config(args),
            )
        )
        print(
            f"Processed {summary.pair_count} pairs; failures={summary.failure_count}; "
            f"manifest={summary.failure_manifest}"
        )
        return 1 if summary.failure_count else 0

    return _dispatch_evaluation(args.evaluation_module, args.evaluation_args)


if __name__ == "__main__":
    raise SystemExit(main())
