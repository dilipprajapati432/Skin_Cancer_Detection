"""
Evaluate a trained run on the test split.

    python evaluate.py --run outputs_strict/densenet121_attn_fusion
    python evaluate.py --run outputs_strict/densenet121_attn_fusion --weights final

Writes metrics.json, classification_report.txt, confusion_matrix.png and
test_predictions.csv into the run folder.
"""
import argparse
import os

import numpy as np

import config


def compute_and_save_metrics(y_true, probs, run_path, class_names, image_ids=None,
                             tag="test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)
    from utils import save_json

    y_pred = probs.argmax(axis=1)
    labels = list(range(len(class_names)))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                  zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                                     zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        auc = float("nan")
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": p, "recall_macro": r, "f1_macro": f1,
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": f1w,
        "roc_auc_macro_ovr": auc,
        "n_samples": int(len(y_true)),
    }
    metrics = {k: float(v) for k, v in metrics.items()}
    save_json(metrics, os.path.join(run_path, f"{tag}_metrics.json"))

    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   digits=4, zero_division=0)
    with open(os.path.join(run_path, f"{tag}_classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, class_names)
    ax.set_yticks(labels, class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({tag}, row-normalised)")
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(run_path, f"{tag}_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    if image_ids is not None:
        out = pd.DataFrame(probs, columns=[f"p_{c}" for c in class_names])
        out.insert(0, "image_id", image_ids)
        out.insert(1, "true", [class_names[i] for i in y_true])
        out.insert(2, "pred", [class_names[i] for i in y_pred])
        out.to_csv(os.path.join(run_path, f"{tag}_predictions.csv"), index=False)

    print("\n" + report)
    print(" | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if k != "n_samples"))
    return metrics


def plot_history(history_csv, run_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    if not os.path.exists(history_csv):
        return
    h = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(h["epoch"] + 1, h[key], label=f"train {key}")
        if f"val_{key}" in h:
            ax.plot(h["epoch"] + 1, h[f"val_{key}"], label=f"val {key}")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_path, "training_curves.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder, e.g. outputs_strict/densenet121_attn_fusion")
    ap.add_argument("--weights", default="best", choices=["best", "final", "latest"])
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = ap.parse_args()

    from data.dataset import build_image_lookup, image_paths_for, make_eval_dataset
    from data.preprocessing import (labels_of, load_and_split_metadata, load_encoder,
                                    transform_clinical)
    from utils import load_trained_model, setup_gpu

    setup_gpu(use_mixed_precision=False)
    model, rc = load_trained_model(args.run, args.weights)
    enc = load_encoder(os.path.join(args.run, "clinical_encoder.json"))

    _, _, test_df = load_and_split_metadata()
    load_images = rc["model_type"] != "clinical_only"
    paths = (image_paths_for(test_df, build_image_lookup()) if load_images
             else np.array([""] * len(test_df)))
    y = labels_of(test_df)
    ds = make_eval_dataset(paths, transform_clinical(test_df, enc), y, args.batch_size,
                           load_images)
    probs = model.predict(ds, verbose=1)
    compute_and_save_metrics(y, probs, args.run, rc["class_names"],
                             image_ids=test_df["image_id"].tolist(),
                             tag=f"test_{args.weights}")


if __name__ == "__main__":
    main()
