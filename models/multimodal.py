"""
Full model from the design sketch:

  Image (224x224x3, 0-255)
    -> Preprocessing (ImageNet normalisation)
    -> Backbone network: DenseNet121 (ImageNet-pretrained)      7 x 7 x 1024
    -> Attention Block (see attention_block.py)                 7 x 7 x 1024
    -> GAP (global average pooling)                             1024
                                                                   \
  Clinical data (age, sex, localization) -> MLP                  64 -> Concat (1088)
                                                                   -> Dense(256) + ReLU + Dropout   (small box)
                                                                   -> FC(7) + softmax               (7 classes)

Also provides image-only and clinical-only variants for the ablation study,
and a `use_attention=False` switch to measure what the attention block adds.
"""
from tensorflow import keras

from models.attention_block import AttentionBlock
from models.backbone import BackbonePreprocess, build_backbone
from models.clinical_mlp import clinical_mlp


def _head(x, num_classes, fusion_units, dropout):
    x = keras.layers.Dense(fusion_units, name="fusion_dense")(x)
    x = keras.layers.BatchNormalization(name="fusion_bn")(x)
    x = keras.layers.Activation("relu", name="fusion_relu")(x)
    x = keras.layers.Dropout(dropout, name="fusion_dropout")(x)
    x = keras.layers.Dense(num_classes, name="fc_logits")(x)
    # float32 output keeps softmax/loss numerically stable under mixed precision
    return keras.layers.Activation("softmax", dtype="float32", name="predictions")(x)


def _image_branch(image_in, backbone_name, img_size, use_attention, weights, dropout,
                  avg_gate="sigmoid"):
    x = BackbonePreprocess(backbone_name, name="preprocess")(image_in)
    backbone = build_backbone(backbone_name, img_size, weights=weights)
    # training=False keeps backbone BatchNorm in inference mode even when fine-tuning
    x = backbone(x, training=False)
    if use_attention:
        x, _attn_map = AttentionBlock(avg_gate=avg_gate, name="attention_block")(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(dropout, name="image_dropout")(x)
    return x


def build_model(model_type, clinical_dim, num_classes=7, backbone="densenet121",
                img_size=224, use_attention=True, mlp_units=(128, 64),
                fusion_units=256, dropout=0.4, weights="imagenet",
                avg_gate="sigmoid"):
    """model_type: 'fusion' (full design) | 'image_only' | 'clinical_only'."""
    image_in = keras.Input(shape=(img_size, img_size, 3), name="image")
    clinical_in = keras.Input(shape=(clinical_dim,), name="clinical")

    if model_type == "fusion":
        img_feat = _image_branch(image_in, backbone, img_size, use_attention, weights, dropout,
                                 avg_gate)
        clin_feat = clinical_mlp(clinical_in, mlp_units, dropout=0.3)
        fused = keras.layers.Concatenate(name="concat")([img_feat, clin_feat])
        out = _head(fused, num_classes, fusion_units, dropout)
        name = f"{backbone}_{'attn' if use_attention else 'noattn'}_fusion"
    elif model_type == "image_only":
        img_feat = _image_branch(image_in, backbone, img_size, use_attention, weights, dropout,
                                 avg_gate)
        out = _head(img_feat, num_classes, fusion_units, dropout)
        name = f"{backbone}_{'attn' if use_attention else 'noattn'}_image_only"
    elif model_type == "clinical_only":
        clin_feat = clinical_mlp(clinical_in, mlp_units, dropout=0.3)
        out = _head(clin_feat, num_classes, fusion_units, dropout)
        name = "clinical_only"
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    # Always take both inputs so every variant uses the same data pipeline.
    return keras.Model(inputs={"image": image_in, "clinical": clinical_in},
                       outputs=out, name=name)


def get_backbone(model):
    """Find the nested backbone model inside a built model (None for clinical_only)."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    return None


if __name__ == "__main__":
    m = build_model("fusion", clinical_dim=20, weights=None)
    m.summary(line_length=110)
