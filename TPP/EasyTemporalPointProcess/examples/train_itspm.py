"""Train ITSPM on Taxi, select by validation likelihood, then test once.

Modified from the EasyTPP example:
resolve local paths and reload the validation-best checkpoint before testing.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner import Runner
from easy_tpp.utils import RunnerPhase
from taxi_config import DATASET_DEFAULTS, FINAL_GRID_BY_DATASET, make_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "taxi")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "taxi")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be cpu, cuda or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU runtime or pass --device cpu for a smoke test.")
    params = copy.deepcopy(DATASET_DEFAULTS["taxi"])
    params.update(FINAL_GRID_BY_DATASET["taxi"][0])
    params.update(gpu=-1 if device.type == "cpu" else device.index,
                  max_epoch=args.epochs, batch_size=args.batch_size)
    cfg = make_config("taxi", params, args.seed, "taxi", False)
    cfg["ITSPM_train"]["base_config"]["base_dir"] = str(args.output_dir.resolve())
    for key, filename in (("train_dir", "train.pkl"), ("valid_dir", "dev.pkl")):
        path = args.data_dir.resolve() / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        cfg["data"]["taxi"][key] = str(path)
    test_path = args.data_dir.resolve() / "test.pkl"
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    config = RunnerConfig.parse_from_yaml_config(cfg, experiment_id="ITSPM_train")
    runner = Runner.build_from_config(config)
    runner.run()
    checkpoint = Path(runner.get_model_dir())
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    runner.model.load_state_dict(state, strict=True)
    runner.runner_config.data_config.test_dir = str(test_path)
    metrics = runner.run_one_epoch(runner._data_loader.test_loader(), RunnerPhase.VALIDATE)
    report = {
        "dataset": "taxi", "seed": args.seed, "epochs": args.epochs,
        "checkpoint": str(checkpoint), "selection": "maximum_validation_loglike",
        "best_epoch_zero_based": runner.metrics_tracker.episode_best,
        "best_validation_loglike": float(runner.metrics_tracker.current_best["loglike"]),
        "test": {key: float(value) for key, value in metrics.items()},
        "is_smoke_test": device.type == "cpu" or args.epochs < 20,
    }
    destination = checkpoint.parent.parent / "test_metrics.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
