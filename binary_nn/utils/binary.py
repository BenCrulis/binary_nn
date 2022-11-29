import numpy as np


def binarize_old(x):
    xb = np.multiply(x > 0, np.int8(2)) - np.int8(1)
    return xb


def binarize(x, out=None, dtype=np.int16):
    if out is None:
        out = np.empty(x.shape, dtype=dtype)
    xb = np.greater(x, 0, out=out)
    xb *= 2
    xb -= 1
    return xb


def binarize01(x):
    xb = x > 0
    return xb


def binary_kwta(x, k):
    if isinstance(k, float):
        k = int(k * x.shape[-1])
    out = np.zeros(x.shape, dtype=np.int8)
    I = np.argsort(x, axis=-1)[:, ::-1]
    np.put_along_axis(out, I[:, :k], 1, axis=-1)
    return out


def kwta(x, k):
    if isinstance(k, float):
        k = int(k * x.shape[-1])
    out = x.copy()
    I = np.argsort(x, axis=-1)[:, ::-1]
    np.put_along_axis(out, I[:, k:], 0, axis=-1)
    return out


def to_bin_targets(y, expand=1, n_classes=None):
    if not n_classes:
        n_classes = len(np.unique(y))
    eye = np.eye(n_classes, dtype=int) * 2 - 1
    oh = eye[y]
    expanded = np.repeat(oh, expand, axis=-1)
    return expanded


def onehot(x, num_classes):
    eye = np.eye(num_classes, dtype="int8")
    return eye[x]


def update_with_prob(a, b, prob):
    return np.where(np.random.choice([True, False], size=a.shape, p=[1.0-prob, prob]), a, b)


def update_col_with_prob(a, b, prob):
    return np.where(np.random.choice([True, False], size=a.shape[-1], p=[1.0-prob, prob]), a.T, b.T).T


def update_fixed_fraction(a, b, fraction, fraction_of_n_errors=False):
    if fraction >= 1.0:
        return b.copy()
    diffs = a != b
    possible_changes = diffs.sum()
    number_of_changes = fraction * possible_changes if fraction_of_n_errors else fraction * b.size
    whole_part = int(np.floor(number_of_changes))
    remainder = number_of_changes - whole_part
    n_changes = whole_part + (1 if np.random.random() < remainder else 0)
    if n_changes >= possible_changes:
        return b.copy()
    idx = np.nonzero(diffs.flat)[0]
    chosen = np.random.choice(idx, size=n_changes, replace=False)
    out = a.copy()
    out.flat[chosen] = b.flat[chosen]
    return out


def update_fixed_fraction_col(a, b, fractions):
    out = []
    for i in range(a.shape[-1]):
        frac = fractions[i] if not isinstance(fractions, float) else fractions
        updated = update_fixed_fraction(a[:, i], b[:, i], frac)
        out.append(updated)
    return np.column_stack(out)


def add_binary_noise(a, prob_noise):
    return a * np.random.choice([-1, 1], size=a.shape, p=[prob_noise, 1.0-prob_noise])


def add_balancing_noise(a, compensation=1.0):
    next_a = a.copy()
    c = a.sum(0)
    nb = (np.abs(c) * compensation / 2.0).astype(int)
    for i in range(a.shape[-1]):
        if c[i] > 0:
            ind = np.nonzero(a[:, i] > 0)[0]
            mod = np.random.choice(ind, size=nb[i], replace=False)
            next_a[mod, i] *= -1
        elif c[i] < 0:
            ind = np.nonzero(a[:, i] < 0)[0]
            mod = np.random.choice(ind, size=nb[i], replace=False)
            next_a[mod, i] *= -1
    return next_a


def random_column_inversion(a, prob):
    return a * np.random.choice([-1, 1], size=(1, a.shape[1]), p=[prob, 1.0-prob])


def random_column_const(a, prob, c=-1):
    b = a.copy()
    dropped = np.random.choice([True, False], size=b.shape[1], p=[prob, 1.0-prob])
    b[:, dropped] = c
    return b


def binary_accuracy(y, y_pred, axis=None):
    return (y == y_pred).sum(axis) / (y_pred.size if axis is None else y_pred.shape[axis])


def balanced_accuracy(y, y_pred, axis=None):
    acc_1, acc_m1 = acc_1_m1(y, y_pred, axis=axis)
    return (acc_1 + acc_m1)/2.0


def acc_1_m1(y, y_pred, axis=None):
    y_pred_1 = y_pred == 1
    y_pred_m1 = y_pred != 1
    denom_1 = y_pred_1.sum(axis=axis)
    if not isinstance(denom_1, np.ndarray):
        if denom_1 == 0:
            denom_1 = 1
    else:
        denom_1[denom_1 == 0] = 1
    acc_1 = ((y == 1) & y_pred_1).sum(axis=axis) / denom_1
    denom_m1 = y_pred_m1.sum(axis=axis)
    if not isinstance(denom_m1, np.ndarray):
        if denom_m1 == 0:
            denom_m1 = 1
    else:
        denom_m1[denom_m1 == 0] = 1
    acc_m1 = ((y != 1) & y_pred_m1).sum(axis=axis) / denom_m1
    return acc_1, acc_m1


def hamming_hsic(x, y):
    K = (x[:, :, None] == x.T).sum(1) / x.shape[-1]
    L = (y[:, :, None] == y.T).sum(1) / y.shape[-1]
    H = np.eye(x.shape[0]) - 1.0/x.shape[0]
    t = np.trace(K @ H @ L @ H)
    crit = t/np.power(x.shape[0]-1, 2.0)
    return crit
