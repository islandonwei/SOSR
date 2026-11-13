"""Portable entry point for the archived, validation-selected HSM trainer."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["hsm", "no-level", "random-neg", "no-contrastive"], default="hsm")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument("--data", type=Path, default=ROOT / "data/semantic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage1-epochs", type=int)
    parser.add_argument("--stage2-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs" / (args.variant + ".json")).read_text())
    for key in ["stage1_epochs", "stage2_epochs", "batch_size"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.variant == "no-level":
        config["stage1_epochs"] = 0
    if config["stage2_epochs"] < 1 or config["batch_size"] < 1 or config["stage1_epochs"] < 0:
        parser.error("Epochs and batch size are invalid")
    data, output = args.data.resolve(), args.output.resolve()
    model = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    for name in ["train.json", "val.json", "categories.csv"]:
        if not (data / name).is_file():
            parser.error(f"Missing {name} in {data}")
    config.update(model=model, data=str(data), output=str(output), device=args.device)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory to preserve prior runs")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["SOSR_ENCODER"] = model
    os.chdir(output)
    sys.path.insert(0, str(ROOT / "hsm/legacy"))
    import torch
    import train as legacy
    import train_ablation as runner
    legacy.config.device = torch.device(args.device)
    legacy.config.stage1_epochs = config["stage1_epochs"]
    legacy.config.stage2_epochs = config["stage2_epochs"]
    legacy.config.stage1_batch_size = legacy.config.stage2_batch_size = config["batch_size"]
    selected_variant = args.variant if args.variant in ["no-level", "random-neg"] else "lambda-1.0"
    runner.arguments = lambda: SimpleNamespace(
        split_dir=str(data), variant=selected_variant, semantic_temperature=0.01,
        pooling="mean", seed=42, encoder_lr=1e-5,
    )
    if args.variant == "no-contrastive":
        from ablation_losses import NoContrastive
        component = NoContrastive()
        runner.base.make_contrastive_loss = lambda *unused: component
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    runner.main()


if __name__ == "__main__":
    main()
