"""
Clinical-data branch ("MLP" box in the design).

Input : one-hot sex + one-hot localization + standardised age (+ dx_type in paper mode)
Output: 64-dim clinical embedding (last value of config.CLINICAL_MLP_UNITS)
"""
from tensorflow import keras


def clinical_mlp(x, units=(128, 64), dropout=0.3, prefix="clinical"):
    """Functional MLP: [Dense -> BatchNorm -> ReLU -> Dropout] per hidden size."""
    for i, u in enumerate(units):
        x = keras.layers.Dense(u, name=f"{prefix}_dense{i + 1}")(x)
        x = keras.layers.BatchNormalization(name=f"{prefix}_bn{i + 1}")(x)
        x = keras.layers.Activation("relu", name=f"{prefix}_relu{i + 1}")(x)
        if i < len(units) - 1:
            x = keras.layers.Dropout(dropout, name=f"{prefix}_drop{i + 1}")(x)
    return x


if __name__ == "__main__":
    inp = keras.Input(shape=(20,))
    m = keras.Model(inp, clinical_mlp(inp))
    print("Output shape:", m.output_shape, " (expect (None, 64))")
