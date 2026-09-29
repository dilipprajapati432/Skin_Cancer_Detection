"""Small shared helpers: GPU setup, run folders, rebuilding a trained model."""
import json
import os

import tensorflow as tf
from tensorflow import keras

import config

# Keras 3 (TF >= 2.16) only saves weights as ".weights.h5".
# Keras 2 (TF 2.10 - 2.15) must use the TF-checkpoint format instead: its .h5
# weight order depends on which layers are frozen, so a checkpoint saved during
# fine-tuning would not load into a freshly built model.
KERAS3 = int(keras.__version__.split(".")[0]) >= 3


def weights_file(run_path, which):
    """Path for 'best' / 'final' / 'latest' weights in the right format."""
    return os.path.join(run_path, f"{which}.weights.h5" if KERAS3 else f"{which}.ckpt")


def weights_exist(path):
    return os.path.exists(path) or os.path.exists(path + ".index")


def setup_gpu(use_mixed_precision=True):
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    if gpus:
        print(f"GPU(s): {[g.name for g in gpus]}")
        if use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision: float16 compute / float32 weights")
    else:
        print("No GPU detected by TensorFlow -> running on CPU (slow). See README 'GPU setup'.")
    return bool(gpus)


def run_dir(run_name):
    path = os.path.join(config.OUTPUT_DIR, run_name)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_trained_model(run_path, which="best"):
    """Rebuild the architecture from run_config.json and load saved weights."""
    from models.multimodal import build_model
    rc = load_json(os.path.join(run_path, "run_config.json"))
    model = build_model(rc["model_type"], clinical_dim=rc["clinical_dim"],
                        num_classes=len(rc["class_names"]), backbone=rc["backbone"],
                        img_size=rc["img_size"], use_attention=rc["use_attention"],
                        mlp_units=tuple(rc["mlp_units"]), fusion_units=rc["fusion_units"],
                        dropout=rc["dropout"], weights=None,
                        avg_gate=rc.get("avg_gate", "sigmoid"))
    wpath = weights_file(run_path, which)
    if not weights_exist(wpath):
        wpath = weights_file(run_path, "final")
    status = model.load_weights(wpath)
    if hasattr(status, "expect_partial"):   # Keras 2 TF-checkpoint: ignore optimizer slots
        status.expect_partial()
    print(f"Loaded weights: {wpath}")
    return model, rc
