"""Command-line entry points for preparation, diagnostics, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .data import audit_table, feature_columns, load_bundle, load_legacy_table, require_evaluable, save_bundle
from .metrics import write_run

_AGFL_CONFIG: dict = {}


def _input_options(parser):
    parser.add_argument("--input", type=Path, required=True, help="Prepared bundle directory or original CSV for diagnostics.")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing results are preserved.")
    parser.add_argument("--allow-unverified", action="store_true", help="Explicit diagnostic mode for legacy feature tables.")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow clearly labeled synthetic software controls.")
    parser.add_argument("--seed", type=int, default=2026)


def _classical_options(parser):
    parser.add_argument("--models", nargs="+", default=["logreg", "hist_gb", "random_forest"],
                        help="Prespecified models; dummy is always included. Optional catboost/xgboost require their extras.")
    parser.add_argument("--inner-splits", type=int, default=5,
                        help="Tuning folds inside each outer training fold (default 5).")
    parser.add_argument("--n-jobs", type=int, default=1)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("alignment-template", help="List original trial IDs with blank, required onset fields.")
    template.add_argument("--dataset-root", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True, help="Output CSV file.")
    template.add_argument("--subjects", nargs="+")
    prepare = commands.add_parser("prepare", help="Prepare EEG and physiology from raw ds004511 using checked trial mappings.")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--alignment", type=Path, help="Checked Subject_ID/Trial_ID/onset/Event_Phase CSV.")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--subjects", nargs="+")
    prepare.add_argument("--tmin", type=float, default=-.2)
    prepare.add_argument("--tmax", type=float, default=1.5)
    prepare.add_argument("--sfreq", type=float, default=256.)
    prepare.add_argument("--include-behavior", action="store_true", help="Include RT predictors available after all trial responses.")
    features = commands.add_parser("features", help="Re-extract features from an aligned bundle.")
    features.add_argument("--input", type=Path, required=True)
    features.add_argument("--output", type=Path, required=True)
    features.add_argument("--include-behavior", action="store_true")
    audit = commands.add_parser("audit", help="Inspect table integrity, predictors and provenance without training.")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path, help="Optional JSON file.")
    benchmark = commands.add_parser("benchmark", help="Nested grouped tabular evaluation, including a majority baseline.")
    _input_options(benchmark)
    _classical_options(benchmark)
    benchmark.add_argument("--feature-sets", nargs="+", help="Default: available modalities and explicit fusion combinations.")
    benchmark.add_argument("--pca-components", type=float, help="Fraction of training variance, strictly between 0 and 1.")
    benchmark.add_argument("--thresholds", nargs="+", type=float, default=[.5], help="Prespecified inner-validation threshold candidates.")
    benchmark.add_argument("--grouping", choices=["loso", "within"], default="loso",
                           help="loso = leave-one-subject-out; within = stratified folds inside each participant.")
    benchmark.add_argument("--per-modality-scaling", action="store_true",
                           help="Impute and standardise each modality block separately inside training folds.")
    benchmark.add_argument("--vector-groups", action="store_true",
                           help="Step 2 of the pipeline note: split EEG into 4 sub-windows x 4 bands and project each group.")
    benchmark.add_argument("--vector-width", type=int, default=16,
                           help="Common vector width for each feature group (default 16).")
    benchmark.add_argument("--equal-width-vectors", action="store_true",
                           help="Pad narrow groups (physiology) to the common width; EEG groups use PCA.")
    agfl = commands.add_parser("agfl", help="AGFL layer comparison: none / k0 / intra / inter / combined.")
    _input_options(agfl)
    agfl.add_argument("--grouping", choices=["loso", "within"], default="within")
    agfl.add_argument("--configs", nargs="+", default=["none", "k0", "intra", "inter", "combined"],
                      choices=["none", "k0", "intra", "inter", "combined"])
    agfl.add_argument("--seeds", nargs="+", type=int, default=[2026])
    agfl.add_argument("--d", type=int, default=16)
    agfl.add_argument("--a", type=int, default=16)
    agfl.add_argument("--hops", type=int, default=2)
    agfl.add_argument("--kappa", type=int, default=5)
    agfl.add_argument("--epochs", type=int, default=60)
    agfl.add_argument("--lr", type=float, default=1e-3)
    agfl.add_argument("--weight-decay", type=float, default=1e-2)
    agfl.add_argument("--shuffle-labels", action="store_true")
    agfl.add_argument("--n-threads", type=int, default=4)
    spatial = commands.add_parser("spatial", help="CSP or xDAWN fitted independently in every training fold.")
    _input_options(spatial)
    _classical_options(spatial)
    spatial.add_argument("--representation", choices=["csp", "xdawn"], default="csp")
    fusion = commands.add_parser("fusion", help="Equal-weight late fusion or grouped out-of-fold stacking.")
    _input_options(fusion)
    fusion.add_argument("--method", choices=["late", "stacking"], default="late")
    fusion.add_argument("--feature-sets", nargs="+", default=["eeg", "physiology"])
    fusion.add_argument("--base-model", default="logreg")
    fusion.add_argument("--inner-splits", type=int, default=3)
    fusion.add_argument("--n-jobs", type=int, default=1)
    neural = commands.add_parser("neural", help="Participant-held-out attention, contrastive, or distillation comparisons.")
    _input_options(neural)
    neural.add_argument("--method", choices=["attention", "contrastive", "distillation"], default="attention")
    neural.add_argument("--secondary", choices=["behavior", "physiology"], default="physiology")
    neural.add_argument("--epochs", type=int, default=40)
    neural.add_argument("--batch-size", type=int, default=64)
    neural.add_argument("--learning-rate", type=float, default=.001)
    llm = commands.add_parser("llm", help="Optional live-service baseline; sends selected numeric features to the configured API.")
    _input_options(llm)
    llm.add_argument("--model", required=True)
    llm.add_argument("--mode", choices=["direct", "few_shot"], default="direct")
    llm.add_argument("--base-url")
    llm.add_argument("--api-key-env", default="OPENAI_API_KEY")
    llm.add_argument("--max-trials", type=int, help="Optional diagnostic sample; omitted means all trials.")
    llm.add_argument("--token-limit-parameter", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
    llm.add_argument("--max-output-tokens", type=int, default=128)
    llm.add_argument("--temperature", type=float, help="Optional provider-supported temperature; omitted by default.")
    mock = commands.add_parser("mock", help="Generate a labeled synthetic software control, not a research dataset.")
    mock.add_argument("--output", type=Path, required=True)
    mock.add_argument("--subjects", type=int, default=6)
    mock.add_argument("--trials", type=int, default=16)
    mock.add_argument("--seed", type=int, default=2026)
    mock.add_argument("--noise-only", action="store_true")
    return parser


def _load(path):
    return load_bundle(path) if path.is_dir() else load_legacy_table(path)


def _new_directory(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Choose a new, empty output directory to preserve previous artifacts.")


def _available_sets(table):
    available = []
    for name in ("eeg", "physiology", "behavior"):
        try:
            feature_columns(table, name)
            available.append(name)
        except ValueError as error:
            if not str(error).startswith("No eligible"):
                raise
    if "eeg" in available and "physiology" in available:
        available.append("eeg_physiology")
    if "eeg" in available and "behavior" in available:
        available.append("eeg_behavior")
    if all(name in available for name in ("eeg", "physiology", "behavior")):
        available.append("fusion")
    elif len(available) >= 2 and not any(name.startswith("eeg_") for name in available):
        available.append("fusion")
    return available


def run(args):
    if args.command == "alignment-template":
        from .preparation import write_alignment_template
        if args.output.exists():
            raise ValueError("The alignment output already exists; preserve checked mappings by using a new path.")
        table = write_alignment_template(args.dataset_root, args.output, args.subjects)
        print(f"Wrote {len(table)} trial IDs to {args.output}; verified onsets and event phase must be supplied.")
        return
    if args.command == "prepare":
        from .preparation import prepare_dataset
        _new_directory(args.output)
        data = prepare_dataset(args.dataset_root, args.output, args.alignment,
                               tmin=args.tmin, tmax=args.tmax, target_sfreq=args.sfreq,
                               subjects=args.subjects, include_behavior=args.include_behavior)
        print(f"Prepared {len(data.table)} aligned trials from {data.table.Subject_ID.nunique()} participants: {args.output}")
        return
    if args.command == "mock":
        from .synthetic import make_control
        _new_directory(args.output)
        save_bundle(make_control(subjects=args.subjects, trials=args.trials, seed=args.seed,
                                 signal=not args.noise_only), args.output)
        print(f"Synthetic software control saved: {args.output}")
        return
    data = _load(args.input)
    if args.command == "audit":
        result = json.dumps(audit_table(data), indent=2, allow_nan=False)
        if args.output:
            if args.output.exists():
                raise ValueError("Audit output exists; use a new path.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result, encoding="utf-8")
        print(result)
        return
    if args.command == "features":
        from .features import extract_features
        _new_directory(args.output)
        save_bundle(extract_features(data, include_behavior=args.include_behavior), args.output)
        print(f"Features saved: {args.output}")
        return
    _new_directory(args.output)
    status = require_evaluable(data, allow_unverified=args.allow_unverified, allow_synthetic=args.allow_synthetic)
    if args.command == "benchmark":
        from .evaluation import evaluate_classical
        if args.pca_components is not None and not 0 < args.pca_components < 1:
            raise ValueError("--pca-components is a training variance fraction strictly between 0 and 1.")
        result = evaluate_classical(data, models=tuple(args.models),
                                    feature_sets=tuple(args.feature_sets or _available_sets(data.table)),
                                    inner_splits=args.inner_splits, seed=args.seed, n_jobs=args.n_jobs,
                                    pca_components=args.pca_components, threshold_grid=tuple(args.thresholds),
                                    grouping=args.grouping,
                                    per_modality_scaling=args.per_modality_scaling,
                                    vector_groups=args.vector_groups,
                                    vector_width=args.vector_width,
                                    equal_width=args.equal_width_vectors)
    elif args.command == "agfl":
        from .torch_benchmark import evaluate_agfl
        result = evaluate_agfl(data, grouping=args.grouping, configs=tuple(args.configs),
                               seeds=tuple(args.seeds), d=args.d, a=args.a, hops=args.hops,
                               kappa=args.kappa, epochs=args.epochs, lr=args.lr,
                               weight_decay=args.weight_decay,
                               shuffle_labels=args.shuffle_labels,
                               fold_seed=args.seed, n_threads=args.n_threads)
        args.output.mkdir(parents=True, exist_ok=True)
        _AGFL_CONFIG.clear()
        _AGFL_CONFIG.update({
            "configs": list(args.configs), "seeds": list(args.seeds),
            "d": args.d, "a": args.a, "hops": args.hops, "kappa": args.kappa,
            "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
            "grouping": args.grouping, "fold_seed": args.seed,
            "shuffle_labels": bool(args.shuffle_labels),
            "encoder": "training-fold StandardScaler + PCA(d) for wide groups, zero padding for narrow groups",
            "pca4": "0.95 fitted on training-fold encoder output; AGFL output uses the same basis",
        })
    elif args.command == "spatial":
        from .evaluation import evaluate_spatial
        result = evaluate_spatial(data, representation=args.representation, models=tuple(args.models),
                                  inner_splits=args.inner_splits, seed=args.seed, n_jobs=args.n_jobs)
    elif args.command == "fusion":
        from .evaluation import evaluate_fusion
        result = evaluate_fusion(data, method=args.method, feature_sets=tuple(args.feature_sets),
                                 base_model=args.base_model, inner_splits=args.inner_splits,
                                 seed=args.seed, n_jobs=args.n_jobs)
    elif args.command == "neural":
        from .neural import evaluate_neural
        result = evaluate_neural(data, method=args.method, secondary=args.secondary, epochs=args.epochs,
                                 batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed)
    elif args.command == "llm":
        from .llm import evaluate_llm
        result = evaluate_llm(data, model=args.model, base_url=args.base_url, mode=args.mode,
                              api_key_env=args.api_key_env, max_trials=args.max_trials, seed=args.seed,
                              token_limit_parameter=args.token_limit_parameter,
                              max_output_tokens=args.max_output_tokens, temperature=args.temperature)
        if result.get("diagnostic"):
            status += ":sampled_diagnostic"
    else:
        raise ValueError("Unknown command.")
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_run(result, data, args.output, configuration=configuration, run_status=status)
    if _AGFL_CONFIG:
        (args.output / "agfl_config.json").write_text(
            json.dumps(_AGFL_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")
        _AGFL_CONFIG.clear()
    print(f"Saved predictions, metrics, fold selections and run metadata: {args.output}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError, ImportError, RuntimeError, FloatingPointError) as error:
        parser.exit(2, f"Error: {error}\n")


def legacy_main(command, *defaults):
    """Keep original script names as thin entry points into the shared pipeline."""
    main([command, *defaults, *sys.argv[1:]])
