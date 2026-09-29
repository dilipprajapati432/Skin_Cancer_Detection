"""
Predict a single lesion image + clinical data with a trained run.

    python predict.py --run outputs_strict/densenet121_attn_fusion ^
        --image path\\to\\ISIC_0024306.jpg --age 45 --sex male --localization back

Prints class probabilities and saves <image>_attention.png next to the run.
(dx_type is only needed for runs trained in "paper" split mode: --dx_type histo)
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--sex", default="unknown")
    ap.add_argument("--localization", default="unknown")
    ap.add_argument("--dx_type", default="unknown")
    ap.add_argument("--weights", default="best", choices=["best", "final", "latest"])
    args = ap.parse_args()

    import pandas as pd
    from tensorflow import keras

    from data.dataset import load_image
    from data.preprocessing import load_encoder, transform_clinical
    from utils import load_trained_model, setup_gpu

    setup_gpu(use_mixed_precision=False)
    model, rc = load_trained_model(args.run, args.weights)
    enc = load_encoder(os.path.join(args.run, "clinical_encoder.json"))

    row = pd.DataFrame([{"age": args.age if args.age is not None else np.nan,
                         "sex": args.sex, "localization": args.localization,
                         "dx_type": args.dx_type}])
    clin = transform_clinical(row, enc)
    img = load_image(args.image, rc["img_size"]).numpy()[None]

    probs = model.predict({"image": img, "clinical": clin}, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    print("\nPrediction:")
    for i in order:
        print(f"  {rc['class_names'][i]:6s} {probs[i]:.4f}")

    if rc["use_attention"] and rc["model_type"] != "clinical_only":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tensorflow as tf
        viz = keras.Model(model.inputs, model.get_layer("attention_block").output[1])
        m = viz.predict({"image": img, "clinical": clin}, verbose=0)[0]
        m = tf.image.resize(m, img.shape[1:3]).numpy()[..., 0]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img[0].astype(np.uint8)); ax[0].set_title("input"); ax[0].axis("off")
        ax[1].imshow(img[0].astype(np.uint8)); ax[1].imshow(m, cmap="jet", alpha=0.45)
        ax[1].set_title(f"attention | pred {rc['class_names'][order[0]]} ({probs[order[0]]:.2f})")
        ax[1].axis("off")
        name = os.path.splitext(os.path.basename(args.image))[0]
        out = os.path.join(args.run, f"{name}_attention.png")
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print("Saved", out)


if __name__ == "__main__":
    main()
