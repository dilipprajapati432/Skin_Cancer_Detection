"""
Image backbone: ImageNet-pretrained DenseNet121 (default) or MobileNetV2,
with the backbone-specific preprocessing built into the model.

Images enter the model as float32 RGB in [0, 255]; the Preprocess layer
applies exactly the normalisation each backbone was pretrained with.
"""
import tensorflow as tf
from tensorflow import keras

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BackbonePreprocess(keras.layers.Layer):
    """[0,255] RGB -> backbone input range.
    densenet121 : /255, then ImageNet mean/std ("torch" mode)
    mobilenetv2 : /127.5 - 1
    """

    def __init__(self, backbone="densenet121", **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone

    def call(self, x):
        x = tf.cast(x, self.compute_dtype)
        if self.backbone == "densenet121":
            mean = tf.constant(IMAGENET_MEAN, dtype=x.dtype)
            std = tf.constant(IMAGENET_STD, dtype=x.dtype)
            return (x / 255.0 - mean) / std
        return x / 127.5 - 1.0

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"backbone": self.backbone})
        return cfg


def build_backbone(name="densenet121", img_size=224, weights="imagenet"):
    """Returns a headless Keras model: (img_size, img_size, 3) -> H x W x C feature map.

    DenseNet121 @224 -> 7 x 7 x 1024
    MobileNetV2 @224 -> 7 x 7 x 1280
    """
    shape = (img_size, img_size, 3)
    if name == "densenet121":
        net = keras.applications.DenseNet121(include_top=False, weights=weights,
                                             input_shape=shape)
    elif name == "mobilenetv2":
        net = keras.applications.MobileNetV2(include_top=False, weights=weights,
                                             input_shape=shape)
    else:
        raise ValueError(f"Unknown backbone '{name}' (use densenet121 or mobilenetv2)")
    return net


def set_backbone_trainable(backbone, trainable, last_n_layers=None):
    """Freeze/unfreeze the backbone. BatchNorm layers always stay frozen
    (standard practice when fine-tuning on a small dataset with small batches)."""
    backbone.trainable = trainable
    if not trainable:
        return
    layers = backbone.layers
    cutoff = 0 if last_n_layers is None else max(0, len(layers) - last_n_layers)
    for i, layer in enumerate(layers):
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = i >= cutoff


if __name__ == "__main__":
    net = build_backbone(weights=None)
    print("Backbone output:", net.output_shape, " (expect (None, 7, 7, 1024))")
    print("Layers:", len(net.layers))
