import numpy as np
from minitorch.modules_transfomer import DecoderLM


def load_weights(model: DecoderLM, npz_path: str):
    """
    Load converted weights from npz into a MiniTorch DecoderLM.
    """
    arrays = np.load(npz_path)

    for name, param in model.named_parameters():
        if name not in arrays:
            print(f"SKIP {name} (not in npz)")
            continue
        arr = arrays[name].astype(np.float32)
        if arr.shape != tuple(param.value._tensor._shape):
            print(f"SHAPE MISMATCH {name}: npz={arr.shape} model={tuple(param.value._tensor._shape)}")
            continue
        param.value._tensor._storage[:] = arr.flatten()

    print("Weights loaded.")