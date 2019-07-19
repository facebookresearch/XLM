import sys
import math
import numpy as np
import torch


# load FAISS GPU library if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss
    FAISS_AVAILABLE = hasattr(faiss, 'StandardGpuResources')
except ImportError:
    FAISS_AVAILABLE = False
    sys.stderr.write("FAISS library was not found.\n")


def get_gaussian_keys(n_keys, dim, normalized, seed):
    """
    Generate random Gaussian keys.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_keys, dim)
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def get_uniform_keys(n_keys, dim, normalized, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    X = rng.uniform(-bound, bound, (n_keys, dim))
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def get_slices(dim, head_id):
    """
    Generate slices of hidden dimensions.
    Used when there are multiple heads and/or different set of keys,
    and that there is no query network.
    """
    if head_id == 0:
        return [(0, dim)]
    offset = dim // (2 ** (head_id + 1))
    starts = np.arange(0, dim, offset)
    slices1 = [(x, x + offset) for i, x in enumerate(starts) if i % 2 == 0]
    slices2 = [(x, x + offset) for i, x in enumerate(starts) if i % 2 == 1]
    return slices1 + slices2


def cartesian_product(a, b):
    """
    Compute the batched cartesian product between two matrices.
    Input:
        a: Tensor(n, d1)
        b: Tensor(n, d2)
    Output:
        output: Tensor(n, d1 * d2, 2)
    """
    n1, d1 = a.shape
    n2, d2 = b.shape
    assert n1 == n2
    return torch.cat([
        a.unsqueeze(-1).repeat(1, 1, d2).unsqueeze(-1),
        b.repeat(1, d1).view(n2, d1, d2).unsqueeze(-1)
    ], 3).view(n1, d1 * d2, 2)


def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(x.storage().data_ptr() + x.storage_offset() * 4)


def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(x.storage().data_ptr() + x.storage_offset() * 8)


def get_knn_pytorch(a, b, k, distance='dot_product'):
    """
    Input:
        - matrix of size (m, d) (keys)
        - matrix of size (n, d) (queries)
        - number of nearest neighbors
        - distance metric
    Output:
        - `scores`  matrix of size (n, k) with nearest neighors scores
        - `indices` matrix of size (n, k) with nearest neighors indices
    """
    m, d = a.size()
    n, _ = b.size()
    assert b.size(1) == d
    assert k > 0
    assert distance in ['dot_product', 'cosine', 'l2']

    with torch.no_grad():

        if distance == 'dot_product':
            scores = a.mm(b.t())                                 # (m, n)

        elif distance == 'cosine':
            scores = a.mm(b.t())                                 # (m, n)
            scores /= (a.norm(2, 1)[:, None] + 1e-9)             # (m, n)
            scores /= (b.norm(2, 1)[None, :] + 1e-9)             # (m, n)

        elif distance == 'l2':
            scores = a.mm(b.t())                                 # (m, n)
            scores *= 2                                          # (m, n)
            scores -= (a ** 2).sum(1)[:, None]                   # (m, n)
            scores -= (b ** 2).sum(1)[None, :]                   # (m, n)

        scores, indices = scores.topk(k=k, dim=0, largest=True)  # (k, n)
        scores = scores.t()                                      # (n, k)
        indices = indices.t()                                    # (n, k)

    return scores, indices


def get_knn_faiss(xb, xq, k, distance='dot_product'):
    """
    `metric` can be faiss.METRIC_INNER_PRODUCT or faiss.METRIC_L2
    https://github.com/facebookresearch/faiss/blob/master/gpu/test/test_pytorch_faiss.py
    """
    assert xb.device == xq.device
    assert distance in ['dot_product', 'l2']
    metric = faiss.METRIC_INNER_PRODUCT if distance == 'dot_product' else faiss.METRIC_L2

    xq_ptr = swig_ptr_from_FloatTensor(xq)
    xb_ptr = swig_ptr_from_FloatTensor(xb)

    nq, d1 = xq.size()
    nb, d2 = xb.size()
    assert d1 == d2

    D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)

    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)

    faiss.bruteForceKnn(
        FAISS_RES, metric,
        xb_ptr, nb,
        xq_ptr, nq,
        d1, k, D_ptr, I_ptr
    )

    return D, I


if FAISS_AVAILABLE:
    FAISS_RES = faiss.StandardGpuResources()
    FAISS_RES.setDefaultNullStreamAllDevices()
    FAISS_RES.setTempMemory(1200 * 1024 * 1024)
    get_knn = get_knn_faiss
else:
    sys.stderr.write("FAISS not available. Switching to standard nearest neighbors search implementation.\n")
    get_knn = get_knn_pytorch
