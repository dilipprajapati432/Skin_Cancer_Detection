# Skin Lesion Classification — DenseNet121 + Attention Block + Clinical MLP (TensorFlow)

HAM10000 7-class classifier implementing the hand-drawn architecture:

```
Image ─► Preprocessing ─► DenseNet121 (ImageNet) ─► Attention Block ─► GAP ──┐
                            7×7×1024                 7×7×1024          1024  ├─► Concat ─► Dense(256) ─► FC(7, softmax)
Clinical data (age, sex, localization) ─► MLP (128 → 64) ──────────────── 64 ──┘
```

**Attention Block** (`models/attention_block.py`), input F = H×W×C:

| Branch | Operation | Output |
|---|---|---|
| 1 | Conv 1×1 (C → 1) | H×W×1 |
| 2 | Max-pool across channels | H×W×1 |
| 3 | Avg-pool across channels → reshape 1×HW → sigmoid → reshape → × avg map | H×W×1 |
| fuse | Concat (H×W×3) → Conv 1×1 → sigmoid = attention map M | H×W×1 |
| out | F × M (broadcast over channels) | H×W×C |

Note on the "densenet21" in the sketch: there's no DenseNet21 in Keras, so this uses **DenseNet121**. MobileNetV2, the other backbone in the sketch, is available through `--backbone mobilenetv2`.

---

## 1. Install TensorFlow with GPU (important on Windows)

TensorFlow **2.11 and newer have no GPU support on native Windows**. Also, TensorFlow does
not have a stable release for Python 3.14 yet. Pick one option:

**Option A: native Windows + GPU (TF 2.10, needs Python 3.10).** Uses Miniconda:
```bash
conda create -n skin python=3.10 -y
conda activate skin
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install "tensorflow==2.10.1" "numpy<2" pandas scikit-learn matplotlib notebook
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Option B: WSL2 (Ubuntu) + GPU (latest TF):**
```bash
pip install "tensorflow[and-cuda]" pandas scikit-learn matplotlib notebook
```

**Option C: CPU only (works, but much slower):** use Python 3.10–3.13 and run `pip install -r requirements.txt`.

The code runs unchanged on TF 2.10 (Keras 2) and TF 2.16+ (Keras 3). The weight file
format is chosen automatically (`.ckpt` on Keras 2, `.weights.h5` on Keras 3).

On first run, the ImageNet DenseNet121 weights (~30 MB) download automatically.

## 2. Set your dataset path
Edit `DATA_ROOT` in `config.py`. It should point to the folder that contains `HAM10000_metadata.csv`,
`HAM10000_images_part_1/` and `HAM10000_images_part_2/`.

## 3. Sanity checks (no dataset needed)
```bash
python models/attention_block.py     # (2,7,7,1024) in -> (2,7,7,1024) out + (2,7,7,1) map
python models/backbone.py
python models/multimodal.py          # full model summary
```

## 4. Train
```bash
python train.py --limit 700 --warmup_epochs 1 --finetune_epochs 1   # 2-minute pipeline test
python train.py                                                    # full model (main result)
```
Training has two phases:
1. **Warm-up** (5 epochs, LR 1e-3). DenseNet121 is frozen, and the attention block, clinical MLP and head are trained.
2. **Fine-tune** (up to 30 epochs, LR 1e-4). The last 150 DenseNet layers are unfrozen (BatchNorm stays frozen). This phase also uses ReduceLROnPlateau and early stopping on val_loss.

**Ablation study:** each of these runs gets its own output folder.
```bash
python train.py --no_attention               # effect of the attention block
python train.py --model image_only           # image branch only
python train.py --model clinical_only        # clinical MLP only
python train.py --backbone mobilenetv2       # alternative backbone
python train.py --avg_gate softmax           # variant: softmax over HW in branch 3
```

**Interrupted?** Re-run the same command with `--resume`. Weights are saved every epoch.

**Out of GPU memory?** Lower `BATCH_SIZE` in `config.py` (16 → 8).

## 5. Evaluate, visualise, predict
```bash
python evaluate.py --run outputs_strict/densenet121_attn_fusion
python visualize_attention.py --run outputs_strict/densenet121_attn_fusion --n 14
python predict.py --run outputs_strict/densenet121_attn_fusion --image path\to\img.jpg --age 45 --sex male --localization back
```

Each run folder (`outputs_<mode>/<run_name>/`) contains:
- weights: best / final / latest
- `history.csv` and `training_curves.png`
- `test_metrics.json`: accuracy, balanced accuracy, macro and weighted P/R/F1, macro ROC-AUC
- `test_classification_report.txt`
- `test_confusion_matrix.png`
- `test_predictions.csv`
- `attention_maps.png`

## Key settings (`config.py`)
- `SPLIT_MODE = "strict"` (default) gives a 70/15/15 split grouped by `lesion_id`, and `dx_type` is excluded.
  Use this mode for honest reported numbers. `"paper"` gives a 70/30 image-level split with `dx_type` included,
  but those numbers are inflated: `dx_type` leaks the label, and images of the same lesion end up in both train and test.
- `BALANCE_STRATEGY = "oversample"` streams an equal number of images per class each epoch
  (`TARGET_PER_CLASS`), with fresh augmentation for every copy. The alternative is `"class_weight"`.
- `USE_MIXED_PRECISION = True` gives float16 compute on the GPU (faster, and uses about half the VRAM).
- `CACHE_IMAGES_IN_RAM = True` makes epochs faster after the first. It needs about 1.5 GB of RAM.

## Project structure
```
config.py                 settings (edit DATA_ROOT)
train.py                  two-phase training + test evaluation
evaluate.py               metrics, confusion matrix, curves
visualize_attention.py    attention-map overlays on test images
predict.py                single image + clinical data -> prediction + attention map
utils.py                  GPU setup, weight saving/loading
data/preprocessing.py     split, clinical encoding (fit on train only)
data/dataset.py           tf.data pipeline, augmentation, class balancing
models/backbone.py        DenseNet121 / MobileNetV2 + ImageNet preprocessing
models/attention_block.py the attention block from the sketch
models/clinical_mlp.py    clinical-data MLP
models/multimodal.py      full model + image_only / clinical_only variants
```
