import numpy as np


def pred_all(f, x):
    res = []
    for line in x:
        res.append(f(line))

    return np.stack(res, axis=0)


def accuracy(pred, true):
    assert len(pred) == len(true)
    return (pred == true).sum() / len(true)


def evaluate(f, x, y_true):
    y_pred = pred_all(f, x)
    return accuracy(y_pred, y_true)


def mse(y_pred, y_true):
    return np.square((y_pred - y_true)).sum()/len(y_pred)


def expanded_pred(y_pred, y_true, expansion):
    y_true_prob = expanded_to_probs(y_true, expansion)
    y_pred_prob = expanded_to_probs(y_pred, expansion)
    return mse(y_pred_prob, y_true_prob)

    y_true_cls = np.argmax(y_true.reshape((y_true.shape[0], -1, expansion)).sum(-1), axis=-1)
    y_pred_cls = np.argmax(y_pred.reshape((y_pred.shape[0], -1, expansion)).sum(-1), axis=-1)
    return accuracy(y_pred_cls, y_true_cls)


def expanded_to_probs(y, expansion):
    y_prob = y.reshape((y.shape[0], -1, expansion)).sum(-1)
    y_prob -= np.min(y_prob, axis=-1)[:, None]
    y_prob = y_prob.astype(float)
    y_sum = np.sum(y_prob, axis=-1)[:, None].astype(float)
    y_sum[y_sum <= 0.0] = 1.0
    y_prob /= y_sum
    return y_prob