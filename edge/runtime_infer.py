import numpy as np

def tanh(x):
    return np.tanh(x)

def dcenn_forward(x, b):
    # x: (F,)
    h = tanh(x @ b["in_W"].T + b["in_b"])
    Wc = b["cell_W"] * b["mask"]
    for _ in range(int(b["steps"][0])):
        h = tanh(h @ Wc.T + b["cell_b"] + 0.3 * h)
    return h

def predict(features_dict, b):
    # features_dict: {feature_name: value}
    feat_names = b["feature_names"].tolist()
    x = np.array([features_dict.get(k, 0.0) for k in feat_names], dtype=np.float32)
    x = (x - b["scaler_mean"]) / (b["scaler_scale"] + 1e-9)
    h = dcenn_forward(x, b)
    y = h @ b["W"]
    # returns vector [CF_Solar, CF_Wind, Load_MW, Price]
    return y
