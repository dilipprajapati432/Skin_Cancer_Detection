"""
Visualise what the Attention Block focuses on.

    python visualize_attention.py --run outputs_strict/densenet121_attn_fusion --n 12

Saves attention_maps.png in the run folder: for each test image the original,
the 7x7 attention map (upsampled) overlaid on it, and true / predicted class.
"""
import argparse
import os

import numpy as np

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", default="best", choices=["best", "final", "latest"])
    ap.add_argument("--n", type=int, default=12, help="number of test images (2 per class-ish)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras

    from data.dataset import build_image_lookup, image_paths_for, load_image
    from data.preprocessing import (labels_of, load_and_split_metadata, load_encoder,
                                    transform_clinical)
    from utils import load_trained_model, setup_gpu

    setup_gpu(use_mixed_precision=False)
    model, rc = load_trained_model(args.run, args.weights)
    if not rc["use_attention"]:
        raise SystemExit("This run was trained without the attention block.")
    enc = load_encoder(os.path.join(args.run, "clinical_encoder.json"))

    attn_layer = model.get_layer("attention_block")
    viz = keras.Model(model.inputs, [attn_layer.output[1], model.output])

    _, _, test_df = load_and_split_metadata()
    # pick a few images from every class
    per = max(1, args.n // config.NUM_CLASSES)
    import pandas as pd
    sample = pd.concat([g.sample(min(len(g), per), random_state=1)
                        for _, g in test_df.groupby("dx")]).reset_index(drop=True)[: args.n]

    paths = image_paths_for(sample, build_image_lookup())
    imgs = np.stack([load_image(p, rc["img_size"]).numpy() for p in paths])
    clin = transform_clinical(sample, enc)
    maps, probs = viz.predict({"image": imgs, "clinical": clin}, verbose=0)
    y = labels_of(sample)

    cols = 4
    rows = int(np.ceil(len(sample) / (cols // 2)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for i in range(len(sample)):
        r, c = divmod(i, cols // 2)
        img = imgs[i].astype(np.uint8)
        m = tf.image.resize(maps[i], img.shape[:2], method="bilinear").numpy()[..., 0]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        pred = int(probs[i].argmax())
        axes[r, 2 * c].imshow(img)
        axes[r, 2 * c].set_title(f"true: {rc['class_names'][y[i]]}", fontsize=9)
        axes[r, 2 * c + 1].imshow(img)
        axes[r, 2 * c + 1].imshow(m, cmap="jet", alpha=0.45)
        ok = "OK" if pred == y[i] else "X"
        axes[r, 2 * c + 1].set_title(f"pred: {rc['class_names'][pred]} "
                                     f"({probs[i, pred]:.2f}) {ok}", fontsize=9)
    fig.suptitle("Attention Block spatial maps (red = high attention)")
    fig.tight_layout()
    out = os.path.join(args.run, "attention_maps.png")
    fig.savefig(out, dpi=130)
    print("Saved", out)


if __name__ == "__main__":
    main()
