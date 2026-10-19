"""Small float64 audit reference; no performance claim is made for this code.

Threshold equality sets the bit. A state's first set bit selects the preceding
upper endpoint; an all-zero field selects the final (lowest) threshold.
"""
import numpy as np
from numbers import Integral


def validate(lut, documents):
    if np.iscomplexobj(lut):
        raise ValueError("Complex LUT values are unsupported")
    lut = np.asarray(lut, dtype=np.float64)
    if lut.ndim != 2 or min(lut.shape) == 0 or not np.isfinite(lut).all():
        raise ValueError('LUT must be a nonempty finite matrix [centroids, terms]')
    if np.max(np.abs(lut)) > np.finfo(np.float64).max / (4*lut.shape[1]):
        raise ValueError("LUT magnitude exceeds the reference arithmetic contract")
    docs = []
    for doc in documents:
        d = np.asarray(doc)
        if d.ndim != 1 or not len(d) or d.dtype.kind not in 'iu':
            raise ValueError('Each document must contain integer code IDs')
        if np.any(d < 0) or np.any(d >= len(lut)):
            raise ValueError('Code ID outside LUT')
        docs.append(d)
    if not docs:
        raise ValueError('At least one document is required')
    return lut, docs


def coarse_scores(lut, documents):
    lut, docs = validate(lut, documents)
    return np.array([lut[d].max(axis=0).sum() for d in docs])


def independent_upper(lut, documents, levels=4, prefix=512):
    lut, docs = validate(lut, documents)
    if (not isinstance(levels, Integral) or levels not in (2, 4, 8)
            or isinstance(prefix, bool) or not isinstance(prefix, Integral) or prefix < 1):
        raise ValueError('levels must be 2, 4 or 8; prefix must be positive')
    prefix = min(prefix, len(lut))
    top = np.sort(lut, axis=0)[::-1][:prefix]
    fractions = np.linspace(1, 0, levels+1)[1:]**2
    thresholds = top[-1] + (top[0]-top[-1])*fractions[:, None]
    table = np.vstack([top[0], thresholds])
    bits = (lut[:, None, :] >= thresholds[None, :, :]).astype(np.uint64)
    code_bits = np.sum(bits << np.arange(levels, dtype=np.uint64)[None, :, None], axis=1)
    upper, packed = [], []
    terms = lut.shape[1]
    for doc in docs:
        fields = np.bitwise_or.reduce(code_bits[doc], axis=0)
        first = [((int(x) & -int(x)).bit_length()-1) if x else levels for x in fields]
        total = table[first, np.arange(terms)].sum()
        # One outward step handles the reference reduction's last rounding.
        upper.append(np.nextafter(total, np.inf))
        value = sum(int(x) << (i*levels) for i, x in enumerate(fields))
        packed.append(value.to_bytes((terms*levels+7)//8, 'little'))
    return np.asarray(upper), packed


def audit(true_scores, upper, budget, selected=None):
    if isinstance(budget, bool) or not isinstance(budget, Integral):
        raise ValueError("budget must be an integer")
    truth, bounds = np.asarray(true_scores), np.asarray(upper)
    if np.iscomplexobj(truth) or np.iscomplexobj(bounds):
        raise ValueError("Complex scores are unsupported")
    if truth.ndim != 1 or truth.shape != bounds.shape or not 1 <= budget <= len(truth):
        raise ValueError('Incompatible scores, bounds or budget')
    if not np.isfinite(truth).all() or not np.isfinite(bounds).all():
        raise ValueError('Nonfinite values require an explicit separate policy')
    cutoff = np.partition(truth, len(truth)-budget)[-budget]
    result = {'threshold': float(cutoff), 'upper_violations': int(np.sum(bounds < truth)),
              'false_survivors': int(np.sum((bounds >= cutoff) & (truth < cutoff))),
              'mean_slack': float(np.mean(bounds-truth)),
              'tie_rule': 'survive at upper >= cutoff; mandatory items have truth > cutoff'}
    if selected is not None:
        selected = np.asarray(selected)
        if selected.ndim != 1 or selected.dtype.kind not in 'iu' or len(selected) != budget:
            raise ValueError('Selected IDs must be an integer vector of length budget')
        if len(np.unique(selected)) != budget or np.any(selected < 0) or np.any(selected >= len(truth)):
            raise ValueError('Selected IDs must be distinct and in range')
        mandatory = set(np.flatnonzero(truth > cutoff))
        result['strict_missing'] = len(mandatory-set(selected.tolist()))
        result['selected_below_cutoff'] = int(np.sum(truth[selected] < cutoff))
    return result
