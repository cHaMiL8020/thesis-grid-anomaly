"""Minimal runtime for dCeNN+ELM bundle on edge devices.

Usage:
    import numpy as np
    from runtime_infer import load_bundle, predict, predict_dict

    bundle = load_bundle("model_bundle.npz")
    features = {
        "Actual_Load_MW":  5000.0,
        "Solar_MW":        200.0,
        "Wind_MW":         800.0,
        # ...
    }

    y = predict(features, bundle)       # np.ndarray aligned with target_names
    y_dict = predict_dict(features, bundle)

"""


import numpy as np


def load_bundle(path: str = "model_bundle.npz"):
    """Load the edge model bundle npz as a dict-like object."""
    return np.load(path, allow_pickle=True)


def tanh(x):
    return np.tanh(x)


def dcenn_forward(x, b):
    """Forward pass of Tiny dCeNN encoder.

    Args:
        x: np.ndarray of shape (F,)
        b: np.load bundle with keys:
           - in_W, in_b
           - cell_W, cell_b
           - mask
           - steps

    Returns:
        h: np.ndarray of shape (enc_dim,)
    """
    in_W = b["in_W"]
    in_b = b["in_b"]
    cell_W = b["cell_W"]
    cell_b = b["cell_b"]
    mask = b["mask"]
    steps = int(b["steps"][0])

    h = tanh(x @ in_W.T + in_b)
    Wc = cell_W * mask
    for _ in range(steps):
        h = tanh(h @ Wc.T + cell_b + 0.3 * h)
    return h


def _build_feature_vector(features_dict, b):
    """Build standardized feature vector from dict and bundle metadata."""
    feat_names = b["feature_names"].tolist()
    x = np.array([features_dict.get(k, 0.0) for k in feat_names], dtype=np.float32)

    scaler_mean = b["scaler_mean"]
    scaler_scale = b["scaler_scale"]
    x = (x - scaler_mean) / (scaler_scale + 1e-9)
    return x


def predict(features_dict, b):
    """Predict targets given a feature dict and loaded bundle.

    Args:
        features_dict: mapping {feature_name: value}
        b: loaded bundle from np.load("model_bundle.npz", allow_pickle=True)

    Returns:
        np.ndarray of shape (num_targets,) aligned with b["target_names"].
    """
    x = _build_feature_vector(features_dict, b)
    h = dcenn_forward(x, b)
    W = b["W"]
    y = h @ W
    return y


def predict_dict(features_dict, b):
    """Predict and return a dict {target_name: value}."""
    y = predict(features_dict, b)
    target_names = b["target_names"].tolist()
    return {name: float(val) for name, val in zip(target_names, y)}
