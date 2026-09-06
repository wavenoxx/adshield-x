"""
AdShield-X :: autodiff.py
-------------------------
A ~200-line reverse-mode automatic-differentiation engine on top of NumPy.

Why it is here: the deep models in this project (DNN / CNN / RNN / CNN+Attention)
are small tabular networks. Shipping them on this micro-engine means the whole
repository trains and reproduces on a plain CPU with `pip install numpy pandas
scikit-learn` and nothing else -- no CUDA, no 600 MB framework download.
`src/keras_models.py` contains the drop-in TensorFlow/Keras equivalents for
anyone who prefers the standard stack; both produce the same architectures.
"""

from __future__ import annotations
import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad")

    def __init__(self, data, _prev=(), requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_prev)
        self.requires_grad = requires_grad

    # ---- pickling ---------------------------------------------------------
    # the autograd graph (closures + parent set) is training-time scaffolding;
    # only the values need to survive a save/load round trip
    def __getstate__(self):
        return {"data": self.data, "requires_grad": self.requires_grad}

    def __setstate__(self, st):
        self.data = st["data"]
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set()
        self.requires_grad = st.get("requires_grad", False)

    # ---- helpers ----------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def _wrap(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    @staticmethod
    def _unbroadcast(g, shape):
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and g.shape[i] != 1:
                g = g.sum(axis=i, keepdims=True)
        return g.reshape(shape)

    # ---- ops --------------------------------------------------------------
    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data, (self, other))

        def _b():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)
        out._backward = _b
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data, (self, other))

        def _b():
            self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += self._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _b
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other))

        def _b():
            self.grad += out.grad @ np.swapaxes(other.data, -1, -2)
            g = np.swapaxes(self.data, -1, -2) @ out.grad
            other.grad += self._unbroadcast(g, other.data.shape)
        out._backward = _b
        return out

    __matmul__ = matmul

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (self._wrap(other) * -1.0)

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,))

        def _b():
            self.grad += out.grad * (out.data > 0)
        out._backward = _b
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))

        def _b():
            self.grad += out.grad * (1 - t * t)
        out._backward = _b
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _b():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.data.shape).copy()
        out._backward = _b
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def max_pool_last(self):
        """Max over the time/length axis (-2) of a (B, L, C) tensor -> (B, C)."""
        idx = np.argmax(self.data, axis=-2)
        B, L, C = self.data.shape
        out_data = np.max(self.data, axis=-2)
        out = Tensor(out_data, (self,))

        def _b():
            g = np.zeros_like(self.data)
            bb = np.arange(B)[:, None]
            cc = np.arange(C)[None, :]
            g[bb, idx, cc] += out.grad
            self.grad += g
        out._backward = _b
        return out

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(p, (self,))

        def _b():
            dot = (out.grad * p).sum(axis=axis, keepdims=True)
            self.grad += p * (out.grad - dot)
        out._backward = _b
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))

        def _b():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _b
        return out

    def bce_with_logits(self, target):
        """Numerically-stable binary cross entropy. target: (B,1) in {0,1}."""
        z = self.data
        t = np.asarray(target, dtype=np.float32).reshape(z.shape)
        loss = np.mean(np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z))))
        out = Tensor(loss, (self,))

        def _b():
            p = 1.0 / (1.0 + np.exp(-z))
            self.grad += out.grad * (p - t) / z.shape[0]
        out._backward = _b
        return out

    # ---- graph ------------------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for c in v._prev:
                build(c)
            topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
def _glorot(fan_in, fan_out, rng):
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, (fan_in, fan_out)).astype(np.float32)


class Dense:
    def __init__(self, nin, nout, rng):
        self.W = Tensor(_glorot(nin, nout, rng), requires_grad=True)
        self.b = Tensor(np.zeros((1, nout)), requires_grad=True)

    def __call__(self, x):
        return x @ self.W + self.b

    def params(self):
        return [self.W, self.b]


class Conv1D:
    """Valid 1-D convolution implemented as an im2col matmul. (B,L,Cin)->(B,L',F)"""

    def __init__(self, cin, filters, k, rng):
        self.k, self.cin, self.f = k, cin, filters
        self.W = Tensor(_glorot(k * cin, filters, rng), requires_grad=True)
        self.b = Tensor(np.zeros((1, filters)), requires_grad=True)

    def __call__(self, x):                      # x: Tensor (B, L, Cin)
        B, L, C = x.shape
        k = self.k
        Lo = L - k + 1
        cols = _im2col(x, k)                    # Tensor (B, Lo, k*C)
        out = cols @ self.W + self.b            # (B, Lo, F)
        return out

    def params(self):
        return [self.W, self.b]


def _im2col(x: Tensor, k: int) -> Tensor:
    B, L, C = x.shape
    Lo = L - k + 1
    idx = np.arange(Lo)[:, None] + np.arange(k)[None, :]
    data = x.data[:, idx, :].reshape(B, Lo, k * C)
    out = Tensor(data, (x,))

    def _b():
        g = out.grad.reshape(B, Lo, k, C)
        acc = np.zeros_like(x.data)
        for j in range(k):
            acc[:, j:j + Lo, :] += g[:, :, j, :]
        x.grad += acc
    out._backward = _b
    return out


class SimpleRNN:
    """tanh recurrent layer returning the last hidden state. (B,L,Cin)->(B,H)"""

    def __init__(self, cin, hidden, rng):
        self.h = hidden
        self.Wx = Tensor(_glorot(cin, hidden, rng), requires_grad=True)
        self.Wh = Tensor(_glorot(hidden, hidden, rng) * 0.5, requires_grad=True)
        self.b = Tensor(np.zeros((1, hidden)), requires_grad=True)

    def __call__(self, x):
        B, L, C = x.shape
        h = Tensor(np.zeros((B, self.h), dtype=np.float32))
        for t in range(L):
            xt = _slice_t(x, t)
            h = (xt @ self.Wx + h @ self.Wh + self.b).tanh()
        return h

    def params(self):
        return [self.Wx, self.Wh, self.b]


def _slice_t(x: Tensor, t: int) -> Tensor:
    out = Tensor(x.data[:, t, :], (x,))

    def _b():
        x.grad[:, t, :] += out.grad
    out._backward = _b
    return out


class AdditiveAttention:
    """Bahdanau-style additive attention pooling over the sequence axis.

    score_t = v^T tanh(W h_t + b)   ->   alpha = softmax(score)
    context = sum_t alpha_t * h_t
    Returns (context, alpha) so the attention weights can be surfaced to the
    analyst as an explanation of *which part of the behaviour vector* drove the
    fraud decision.
    """

    def __init__(self, cin, units, rng):
        self.W = Tensor(_glorot(cin, units, rng), requires_grad=True)
        self.b = Tensor(np.zeros((1, units)), requires_grad=True)
        self.v = Tensor(_glorot(units, 1, rng), requires_grad=True)

    def __call__(self, h):                       # h: (B, L, C)
        e = (h @ self.W + self.b).tanh() @ self.v          # (B, L, 1)
        a = e.softmax(axis=1)                              # (B, L, 1)
        ctx = (h * a).sum(axis=1)                          # (B, C)
        return ctx, a

    def params(self):
        return [self.W, self.b, self.v]


class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, wd
        self.m = [np.zeros_like(x.data) for x in params]
        self.v = [np.zeros_like(x.data) for x in params]
        self.t = 0

    def zero_grad(self):
        for x in self.p:
            x.grad = np.zeros_like(x.data)

    def step(self):
        self.t += 1
        for i, x in enumerate(self.p):
            g = x.grad + self.wd * x.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            x.data -= self.lr * mh / (np.sqrt(vh) + self.eps)
