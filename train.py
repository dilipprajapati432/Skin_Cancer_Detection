"""
Train the DenseNet121 + Attention Block + Clinical-MLP fusion model (TensorFlow/Keras).

Two-phase transfer learning:
  Phase 1 (warm-up)  : backbone frozen -> train attention block, clinical MLP and head
  Phase 2 (fine-tune): unfreeze the last FINE_TUNE_LAYERS backbone layers, lower LR

Usage:
    python train.py                                   # full model (fusion + attention, DenseNet121)
    python train.py --model image_only                # ablation: image branch only
    python train.py --model clinical_only             # ablation: clinical MLP only
    python train.py --no_attention                    # ablation: remove the attention block
    python train.py --backbone mobilenetv2            # lighter backbone from the sketch
    python train.py --limit 700 --warmup_epochs 1 --finetune_epochs 1   # quick smoke test
    python train.py --resume                          # continue an interrupted run

Outputs go to  outputs_<split_mode>/<run_name>/
"""
import argparse
import os
import time

import numpy as np

import config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fusion", choices=["fusion", "image_only", "clinical_only"])
    ap.add_argument("--backbone", default=config.BACKBONE, choices=["densenet121", "mobilenetv2"])
    ap.add_argument("--no_attention", action="store_true")
    ap.add_argument("--avg_gate", default="sigmoid", choices=["sigmoid", "softmax"],
                    help="gate used in the attention block's avg-pool branch")
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--warmup_epochs", type=int, default=config.WARMUP_EPOCHS)
    ap.add_argument("--finetune_epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    ap.add_argument("--warmup_lr", type=float, default=config.WARMUP_LR)
    ap.add_argument("--finetune_lr", type=float, default=config.FINE_TUNE_LR)
    ap.add_argument("--finetune_layers", type=int, default=config.FINE_TUNE_LAYERS,
                    help="how many backbone layers to unfreeze in phase 2 (-1 = all)")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only N metadata rows (quick pipeline test)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true",
                    help="random-init backbone (debug only)")
    return ap.parse_args()


def main():
    args = parse_args()

    import tensorflow as tf
    from tensorflow import keras

    from data.dataset import (build_image_lookup, image_paths_for, make_eval_dataset,
                              make_train_dataset)
    from data.preprocessing import (fit_clinical_encoder, labels_of, load_and_split_metadata,
                                    save_encoder, transform_clinical)
    from evaluate import compute_and_save_metrics, plot_history
    from models.backbone import set_backbone_trainable
    from models.multimodal import build_model, get_backbone
    from utils import load_json, run_dir, save_json, setup_gpu, weights_exist, weights_file

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    setup_gpu(config.USE_MIXED_PRECISION)

    # ------------------------------------------------------------ data
    train_df, val_df, test_df = load_and_split_metadata()
    if args.limit:
        frac = min(1.0, args.limit / (len(train_df) + len(test_df) + (len(val_df) if val_df is not None else 0)))
        import pandas as pd
        sub = lambda d: pd.concat(
            [g.sample(min(len(g), max(2, int(len(g) * frac))), random_state=0)
             for _, g in d.groupby("dx")]).reset_index(drop=True)
        train_df, test_df = sub(train_df), sub(test_df)
        val_df = sub(val_df) if val_df is not None else None
        print(f"[limit] train {len(train_df)} / val {0 if val_df is None else len(val_df)} / test {len(test_df)}")

    enc = fit_clinical_encoder(train_df)
    clin = {"train": transform_clinical(train_df, enc), "test": transform_clinical(test_df, enc)}
    y = {"train": labels_of(train_df), "test": labels_of(test_df)}
    if val_df is not None:
        clin["val"], y["val"] = transform_clinical(val_df, enc), labels_of(val_df)
    clinical_dim = clin["train"].shape[1]
    print(f"Clinical features ({clinical_dim}): {enc['feature_names']}")
    print("Train class counts:", dict(zip(config.CLASS_NAMES, np.bincount(y['train'], minlength=7))))

    load_images = args.model != "clinical_only"
    if load_images:
        lookup = build_image_lookup()
        paths = {"train": image_paths_for(train_df, lookup), "test": image_paths_for(test_df, lookup)}
        if val_df is not None:
            paths["val"] = image_paths_for(val_df, lookup)
    else:
        paths = {k: np.array([""] * len(v)) for k, v in y.items()}

    train_ds, steps = make_train_dataset(paths["train"], clin["train"], y["train"],
                                         args.batch_size, load_images)
    val_ds = (make_eval_dataset(paths["val"], clin["val"], y["val"], args.batch_size, load_images)
              if val_df is not None else None)
    test_ds = make_eval_dataset(paths["test"], clin["test"], y["test"], args.batch_size, load_images)

    class_weight = None
    if config.BALANCE_STRATEGY == "class_weight":
        from sklearn.utils.class_weight import compute_class_weight
        present = np.unique(y["train"])
        w = compute_class_weight("balanced", classes=present, y=y["train"])
        class_weight = {int(c): float(v) for c, v in zip(present, w)}
        print("Class weights:", class_weight)

    # ------------------------------------------------------------ model
    use_attention = not args.no_attention
    model = build_model(args.model, clinical_dim, config.NUM_CLASSES, args.backbone,
                        config.IMG_SIZE, use_attention, config.CLINICAL_MLP_UNITS,
                        config.FUSION_UNITS, config.DROPOUT,
                        weights=None if args.no_pretrained else "imagenet",
                        avg_gate=args.avg_gate)
    run_name = args.run_name or model.name
    out = run_dir(run_name)
    print(f"\nRun: {run_name}  ->  {out}")

    save_encoder(enc, os.path.join(out, "clinical_encoder.json"))
    save_json({"model_type": args.model, "backbone": args.backbone,
               "use_attention": use_attention, "avg_gate": args.avg_gate,
               "clinical_dim": clinical_dim, "img_size": config.IMG_SIZE,
               "mlp_units": list(config.CLINICAL_MLP_UNITS), "fusion_units": config.FUSION_UNITS,
               "dropout": config.DROPOUT, "class_names": config.CLASS_NAMES,
               "split_mode": config.SPLIT_MODE, "balance": config.BALANCE_STRATEGY,
               "batch_size": args.batch_size}, os.path.join(out, "run_config.json"))

    backbone = get_backbone(model)
    has_val = val_ds is not None
    monitor = "val_loss" if has_val else "loss"
    state_path = os.path.join(out, "state.json")
    latest_w = weights_file(out, "latest")
    best_w = weights_file(out, "best")
    history_csv = os.path.join(out, "history.csv")

    # --------------------------------------------------- resume support
    state = {"epochs_done": 0, "best": None}
    if args.resume and os.path.exists(state_path) and weights_exist(latest_w):
        state = load_json(state_path)
        st = model.load_weights(latest_w)
        if hasattr(st, "expect_partial"):
            st.expect_partial()
        print(f"Resuming after epoch {state['epochs_done']} "
              f"(optimizer state restarts; weights restored)")
    elif os.path.exists(history_csv):
        os.remove(history_csv)

    class SaveState(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            model.save_weights(latest_w)
            cur = logs.get(monitor)
            if cur is not None and (state["best"] is None or cur < state["best"]):
                state["best"] = float(cur)
            state["epochs_done"] = epoch + 1
            save_json(state, state_path)

    def callbacks(phase_patience=True):
        cbs = [SaveState(), keras.callbacks.CSVLogger(history_csv, append=True)]
        if has_val:
            ckpt_kwargs = dict(monitor="val_loss", save_best_only=True,
                               save_weights_only=True, verbose=1)
            if state["best"] is not None:
                ckpt_kwargs["initial_value_threshold"] = state["best"]
            cbs.append(keras.callbacks.ModelCheckpoint(best_w, **ckpt_kwargs))
            cbs.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                                         patience=3, min_lr=1e-7, verbose=1))
            if phase_patience:
                cbs.append(keras.callbacks.EarlyStopping(monitor="val_loss",
                                                         patience=config.EARLY_STOP_PATIENCE,
                                                         verbose=1))
        return cbs

    def compile_model(lr):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
            metrics=["accuracy", keras.metrics.AUC(name="auc")])

    t0 = time.time()
    fit_kw = dict(steps_per_epoch=steps, validation_data=val_ds,
                  class_weight=class_weight, verbose=1)

    if args.model == "clinical_only":
        total = args.warmup_epochs + args.finetune_epochs
        compile_model(args.warmup_lr)
        model.fit(train_ds, epochs=total, initial_epoch=state["epochs_done"],
                  callbacks=callbacks(), **fit_kw)
    else:
        # ---------- Phase 1: frozen backbone
        W = args.warmup_epochs
        if state["epochs_done"] < W:
            set_backbone_trainable(backbone, False)
            compile_model(args.warmup_lr)
            print(f"\n=== Phase 1: backbone frozen | trainable params "
                  f"{sum(int(np.prod(v.shape)) for v in model.trainable_weights):,}")
            model.fit(train_ds, epochs=W, initial_epoch=state["epochs_done"],
                      callbacks=callbacks(phase_patience=False), **fit_kw)

        # ---------- Phase 2: fine-tune
        n = None if args.finetune_layers is None or args.finetune_layers < 0 else args.finetune_layers
        set_backbone_trainable(backbone, True, last_n_layers=n)
        compile_model(args.finetune_lr)
        print(f"\n=== Phase 2: fine-tuning last {n or 'ALL'} backbone layers | trainable params "
              f"{sum(int(np.prod(v.shape)) for v in model.trainable_weights):,}")
        model.fit(train_ds, epochs=W + args.finetune_epochs,
                  initial_epoch=max(W, state["epochs_done"]), callbacks=callbacks(), **fit_kw)

    model.save_weights(weights_file(out, "final"))
    print(f"\nTraining time: {(time.time() - t0) / 60:.1f} min")
    plot_history(history_csv, out)

    # ------------------------------------------------------------ test
    if has_val and weights_exist(best_w):
        st = model.load_weights(best_w)
        if hasattr(st, "expect_partial"):
            st.expect_partial()
        print("Evaluating BEST-validation weights on the test set")
    else:
        print("Evaluating FINAL weights on the test set")
    probs = model.predict(test_ds, verbose=1)
    compute_and_save_metrics(y["test"], probs, out, config.CLASS_NAMES,
                             image_ids=test_df["image_id"].tolist(), tag="test")
    print(f"\nAll outputs saved in: {out}")


if __name__ == "__main__":
    main()
