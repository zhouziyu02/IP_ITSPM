"""Fixed Taxi experiment configuration, with no search or test-based selection."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET_DEFAULTS = {"taxi": {"batch_size": 128, "max_epoch": 20, "loss_samples": 8}}
FINAL_GRID_BY_DATASET = {"taxi": [{
    "hidden_size": 64, "window_size": 16, "dropout_rate": 0.10,
    "learning_rate": 5e-4, "head_type": "decay_mlp", "n_mixer_layers": 1,
}]}


def make_config(dataset, params, seed, run_id, include_test):
    """Keep the original selected Taxi architecture and optimizer parameters."""
    if dataset != "taxi":
        raise ValueError("This entry point supports the Taxi experiment only.")
    with (ROOT / "examples" / "configs" / "itspm_tpp_config.yaml").open() as stream:
        cfg = yaml.safe_load(stream)
    exp = cfg["ITSPM_train"]
    exp["base_config"]["dataset_id"] = dataset
    exp["base_config"]["base_dir"] = str(ROOT / "outputs" / run_id)
    exp["trainer_config"].update(
        seed=seed, gpu=params.get("gpu", 0), shuffle=True,
        batch_size=params.get("batch_size", DATASET_DEFAULTS[dataset]["batch_size"]),
        max_epoch=params.get("max_epoch", DATASET_DEFAULTS[dataset]["max_epoch"]),
        learning_rate=params["learning_rate"],
    )
    exp["model_config"].update(
        hidden_size=params["hidden_size"], dropout_rate=params["dropout_rate"],
        num_layers=params["n_mixer_layers"], num_heads=2,
        loss_integral_num_sample_per_step=params.get("loss_samples", DATASET_DEFAULTS[dataset]["loss_samples"]),
    )
    exp["model_config"]["model_specs"].update(
        window_size=params["window_size"],
        max_event_tokens=params.get("max_event_tokens", params["window_size"]),
        max_gap_tokens=params.get("max_gap_tokens", max(8, params["window_size"] // 2)),
        n_mixer_layers=params["n_mixer_layers"], n_heads=2,
        head_type=params["head_type"], time_emb_size=min(64, params["hidden_size"]),
    )
    if not include_test:
        cfg["data"][dataset]["test_dir"] = None
    return cfg
