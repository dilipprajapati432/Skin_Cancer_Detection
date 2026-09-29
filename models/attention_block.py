"""
Attention Block -- implemented exactly as drawn in the design sketch.

Input feature map F : H x W x C   (DenseNet121 output: 7 x 7 x 1024 at 224px)

  Branch 1 (learned):   Conv 1x1 (C -> 1)                         -> H x W x 1
  Branch 2 (max):       Max-pool across channels                  -> H x W x 1
  Branch 3 (avg, self-gated):
                        A = Avg-pool across channels               -> H x W x 1
                        reshape A -> 1 x HW, sigmoid, reshape back -> gate
                        A * gate                                   -> H x W x 1

  Concatenate [B1, B2, B3]                                          -> H x W x 3
  Conv 1x1 (3 -> 1)                                                 -> H x W x 1
  Sigmoid                                                           -> H x W x 1   (attention map M)
  Output = F * M   (M broadcast over all C channels)                -> H x W x C

Note on branch 3: sigmoid is element-wise, so the reshape to 1xHW does not
change the result (it is kept to mirror the diagram). If you want the gate to
be a true *spatial distribution* over the HW positions, set
avg_gate="softmax" -- then the reshape matters (softmax is taken over all HW).
"""
import tensorflow as tf
from tensorflow import keras


class AttentionBlock(keras.layers.Layer):
    """Returns [attended_features (HxWxC), attention_map (HxWx1)]."""

    def __init__(self, avg_gate="sigmoid", **kwargs):
        super().__init__(**kwargs)
        if avg_gate not in ("sigmoid", "softmax"):
            raise ValueError("avg_gate must be 'sigmoid' or 'softmax'")
        self.avg_gate = avg_gate
        # Branch 1: 1x1 conv C -> 1
        self.branch_conv = keras.layers.Conv2D(1, kernel_size=1, padding="same",
                                               name="branch_conv1x1")
        # Fusion: 1x1 conv over the 3 stacked descriptors -> 1
        self.fuse_conv = keras.layers.Conv2D(1, kernel_size=1, padding="same",
                                             name="fuse_conv1x1")

    def call(self, x):
        # ---- Branch 1: learned 1x1 conv -> H x W x 1
        b1 = self.branch_conv(x)

        # ---- Branch 2: channel-wise max pool -> H x W x 1
        b2 = tf.reduce_max(x, axis=-1, keepdims=True)

        # ---- Branch 3: channel-wise avg pool, reshaped to 1 x HW, gated, reshaped back
        a = tf.reduce_mean(x, axis=-1, keepdims=True)          # B x H x W x 1
        shape = tf.shape(a)
        a_flat = tf.reshape(a, [shape[0], 1, shape[1] * shape[2]])   # B x 1 x HW
        if self.avg_gate == "sigmoid":
            gate = tf.sigmoid(a_flat)
        else:
            gate = tf.nn.softmax(a_flat, axis=-1)
        gate = tf.reshape(gate, shape)                           # B x H x W x 1
        b3 = a * gate

        # ---- Concatenate -> H x W x 3, conv 1x1 -> H x W x 1, sigmoid
        stacked = tf.concat([b1, b2, b3], axis=-1)
        attn_map = tf.sigmoid(self.fuse_conv(stacked))

        # ---- Re-weight the original feature map (broadcast over C)
        out = x * attn_map
        return [out, attn_map]

    def compute_output_shape(self, input_shape):
        return [tuple(input_shape), tuple(input_shape[:-1]) + (1,)]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"avg_gate": self.avg_gate})
        return cfg


if __name__ == "__main__":
    block = AttentionBlock()
    x = tf.random.normal((2, 7, 7, 1024))
    out, m = block(x)
    print("input :", x.shape)
    print("output:", out.shape, " (expect (2, 7, 7, 1024))")
    print("map   :", m.shape, " (expect (2, 7, 7, 1))")
    print("params:", block.count_params(), " (1024+1 for branch conv, 3+1 for fuse conv = 1029)")
