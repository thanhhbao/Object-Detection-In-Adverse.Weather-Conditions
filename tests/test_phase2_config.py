"""Phase 2 RT-DETR config must resolve to the merged dataset, not XWOD, and
must keep the low-LR / batch-16 continued fine-tune settings."""

from dawn_ablation.common import load_experiment_config


def test_phase2_config_resolves_merged_dataset_and_hparams():
    config = load_experiment_config("configs/ultralytics/phase2_final_rtdetr_from_xwod.yaml")

    assert config["dataset"] == "phase2_merged"
    assert config["from_run"] == "stage2_xwod_rtdetr_from_bdd30k"
    assert config["batch"] == 16
    assert config["lr0"] == 0.00005
    assert config["patience"] == 20

    # Checkpoint must resolve via from_run, never be hard-coded.
    model_path = str(config["model"]).replace("\\", "/")
    assert model_path.endswith("stage2_xwod_rtdetr_from_bdd30k/weights/best.pt")

    # Values not overridden must come from train_defaults.yaml.
    assert config["epochs"] == 50
    assert config["imgsz"] == 640
    assert config["optimizer"] == "AdamW"
    assert config["seed"] == 42
    assert config["deterministic"] is True
    assert config["amp"] is True
    assert config["cos_lr"] is True


def test_stage2_config_unaffected_by_phase2_changes():
    """Stage 2 must stay exactly as before: XWOD dataset, batch inherited from defaults (16)."""
    config = load_experiment_config("configs/ultralytics/stage2_xwod_rtdetr_from_bdd30k.yaml")

    assert config["dataset"] == "xwod"
    assert config["from_run"] == "stage1_bdd30k_rtdetr"
    assert config["lr0"] == 0.0005
    assert config["patience"] == 15
    assert config["batch"] == 16  # inherited from train_defaults.yaml, untouched by Phase 2
