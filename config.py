"""
Central configuration -- HAM10000 multimodal skin-lesion classifier (TensorFlow).

Architecture (from the hand-drawn design):

    Image -> Preprocessing -> Backbone (DenseNet121, ImageNet) -> Attention Block -> GAP --+
                                                                                          (concat) -> Dense -> FC(7)
    Clinical data (age, sex, localization) -> MLP ----------------------------------------+

*** EDIT DATA_ROOT BELOW to match your folder ***
"""
import os

# ---------------- DATA ----------------
# Folder containing HAM10000_metadata.csv, HAM10000_images_part_1/, HAM10000_images_part_2/
DATA_ROOT = r"C:\Users\Dilip Prajapati\Desktop\MInor Project_Skin\Dataset\Data"   # <-- EDIT THIS LINE

METADATA_CSV = os.path.join(DATA_ROOT, "HAM10000_metadata.csv")
IMAGE_DIRS = [
    os.path.join(DATA_ROOT, "HAM10000_images_part_1"),
    os.path.join(DATA_ROOT, "HAM10000_images_part_2"),
]
NUM_CLASSES = 7
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ---------------- IMAGE ----------------
# 224x224 is DenseNet121's native ImageNet resolution and fits a 6GB GPU.
IMG_SIZE = 224

# ---------------- SPLIT MODE ----------------
# "strict" (default, recommended for reporting):
#     70/15/15 train/val/test split GROUPED BY lesion_id (the same lesion never
#     appears in both train and test), dx_type EXCLUDED from clinical features
#     (dx_type nearly reveals the label -> inflated accuracy).
# "paper":
#     70/30 random image-level split, no validation set, dx_type included.
SPLIT_MODE = "strict"

USE_DX_TYPE = SPLIT_MODE == "paper"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.0 if SPLIT_MODE == "paper" else 0.15
TEST_FRAC = 0.30 if SPLIT_MODE == "paper" else 0.15
RANDOM_SEED = 42

# ---------------- CLASS IMBALANCE ----------------
# HAM10000 is ~67% "nv". Pick one strategy:
#   "oversample"   -> replicate/undersample TRAIN rows to TARGET_PER_CLASS each
#                     (fresh random augmentation on every copy)
#   "class_weight" -> keep natural distribution, weight the loss per class
#   "none"
BALANCE_STRATEGY = "oversample"
TARGET_PER_CLASS = 2000      # 7 x 2000 = 14,000 training images / epoch

# ---------------- MODEL ----------------
BACKBONE = "densenet121"     # "densenet121" (default) or "mobilenetv2" (lighter alternative)
USE_ATTENTION = True         # set False (or --no_attention) for the ablation study
CLINICAL_MLP_UNITS = (128, 64)   # clinical MLP hidden sizes -> 64-dim clinical embedding
FUSION_UNITS = 256           # dense layer after concatenation (the small box before FC)
DROPOUT = 0.4

# ---------------- TRAINING (tuned for RTX 3050 6GB) ----------------
BATCH_SIZE = 16
# Phase 1: backbone frozen, train attention + MLP + head
WARMUP_EPOCHS = 5
WARMUP_LR = 1e-3
# Phase 2: unfreeze the backbone (last FINE_TUNE_LAYERS layers) and fine-tune
FINE_TUNE_EPOCHS = 30
FINE_TUNE_LR = 1e-4
FINE_TUNE_LAYERS = 150       # DenseNet121 has ~427 layers; None = unfreeze all
EARLY_STOP_PATIENCE = 8      # only used when a validation set exists
LABEL_SMOOTHING = 0.05

USE_MIXED_PRECISION = True   # float16 compute on GPU: faster + half the VRAM
CACHE_IMAGES_IN_RAM = False  # True = decode each image once (~1.5 GB RAM at 224px)

# ---------------- OUTPUTS ----------------
OUTPUT_DIR = "outputs_" + SPLIT_MODE   # weights, logs, plots, metrics per run
