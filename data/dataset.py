"""
tf.data input pipeline.

Each element: ({"image": float32[IMG, IMG, 3] in 0..255, "clinical": float32[F]}, one_hot[7])

Training set balancing ("oversample" mode) is done by streaming: one shuffled,
repeating dataset per class, sampled with equal probability. Every epoch
therefore sees ~TARGET_PER_CLASS images per class, minority images get a fresh
random augmentation each time they are repeated, and the majority class (nv)
rotates through ALL of its images across epochs instead of being cut once.
"""
import os
import numpy as np
import tensorflow as tf

import config

AUTOTUNE = tf.data.AUTOTUNE


def build_image_lookup(image_dirs=None):
    lookup = {}
    for d in image_dirs or config.IMAGE_DIRS:
        if not os.path.isdir(d):
            print(f"[warn] image folder not found: {d}")
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                lookup[os.path.splitext(fname)[0]] = os.path.join(d, fname)
    return lookup


def image_paths_for(df, lookup):
    missing = [i for i in df["image_id"] if i not in lookup]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[:3]}. "
                                f"Check DATA_ROOT / IMAGE_DIRS in config.py")
    return np.array([lookup[i] for i in df["image_id"]])


# ---------------------------------------------------------------- image ops
def load_image(path, img_size=None):
    img_size = img_size or config.IMG_SIZE
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size), method="bilinear", antialias=True)
    return tf.cast(img, tf.float32)          # 0..255, normalised inside the model


def augment(img):
    """Dermoscopy-safe augmentation: lesions have no canonical orientation."""
    size = tf.shape(img)[0]
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    # random zoom: crop 80-100% of the side, resize back
    scale = tf.random.uniform([], 0.80, 1.0)
    crop = tf.cast(tf.cast(size, tf.float32) * scale, tf.int32)
    img = tf.image.random_crop(img, tf.stack([crop, crop, 3]))
    img = tf.image.resize(img, tf.stack([size, size]))
    # colour jitter (in 0..1 space)
    img = img / 255.0
    img = tf.image.random_brightness(img, 0.10)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    img = tf.image.random_hue(img, 0.02)
    return tf.clip_by_value(img, 0.0, 1.0) * 255.0


# ---------------------------------------------------------------- datasets
def _base(paths, clinical, labels, load_images=True):
    onehot = tf.one_hot(labels, config.NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((paths, clinical, onehot))

    def _load(p, c, y):
        img = load_image(p) if load_images else tf.zeros((config.IMG_SIZE, config.IMG_SIZE, 3))
        return img, c, y
    return ds.map(_load, num_parallel_calls=AUTOTUNE)


def _to_model_inputs(img, c, y):
    return {"image": img, "clinical": c}, y


def make_eval_dataset(paths, clinical, labels, batch_size, load_images=True):
    ds = _base(paths, clinical, labels, load_images)
    if config.CACHE_IMAGES_IN_RAM:
        ds = ds.cache()
    return ds.map(_to_model_inputs).batch(batch_size).prefetch(AUTOTUNE)


def make_train_dataset(paths, clinical, labels, batch_size, load_images=True):
    """Returns (dataset, steps_per_epoch)."""
    def _aug(img, c, y):
        return (augment(img) if load_images else img), c, y

    if config.BALANCE_STRATEGY == "oversample":
        per_class = []
        for k in range(config.NUM_CLASSES):
            m = labels == k
            if not m.any():
                continue
            ds_k = _base(paths[m], clinical[m], labels[m], load_images)
            if config.CACHE_IMAGES_IN_RAM:
                ds_k = ds_k.cache()
            per_class.append(ds_k.shuffle(int(m.sum()), seed=config.RANDOM_SEED + k,
                                          reshuffle_each_iteration=True).repeat())
        weights = [1.0 / len(per_class)] * len(per_class)
        ds = tf.data.Dataset.sample_from_datasets(per_class, weights=weights,
                                                  seed=config.RANDOM_SEED)
        steps = int(np.ceil(config.TARGET_PER_CLASS * len(per_class) / batch_size))
    else:
        ds = _base(paths, clinical, labels, load_images)
        if config.CACHE_IMAGES_IN_RAM:
            ds = ds.cache()
        ds = ds.shuffle(len(labels), seed=config.RANDOM_SEED,
                        reshuffle_each_iteration=True).repeat()
        steps = int(np.ceil(len(labels) / batch_size))

    ds = (ds.map(_aug, num_parallel_calls=AUTOTUNE)
            .map(_to_model_inputs)
            .batch(batch_size, drop_remainder=True)
            .prefetch(AUTOTUNE))
    return ds, steps
