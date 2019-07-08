from kerassurgeon import Surgeon
import numpy as np


def prune(model, thresh=0.1):
    '''
    model: model to prune
    thresh: neurons, corresponding to BSFilter layers lower
    than thresh will be removed
    '''
    bs_idx = []
    for idx, l in enumerate(model.layers):
        if l.__class__.__name__ == "BSFilter":
            bs_idx.append(idx)

    weights = []
    surgeon = Surgeon(model)
    for i in bs_idx:
        weights = model.layers[i].get_weights()[0]
        l = model.layers[i]
        to_prune = model.layers[i - 1]
        surgeon.add_job('delete_layer', l)
        surgeon.add_job('delete_channels', to_prune, channels=np.where(weights < thresh)[0].tolist())
    return surgeon.operate()
