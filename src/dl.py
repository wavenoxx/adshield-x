"""
AdShield-X :: dl.py
-------------------
Deep models with a scikit-learn-style API (fit / predict / predict_proba).

A tabular click record of D features is treated as a length-D sequence with one
channel, which is the standard 1-D-CNN-on-tabular formulation used by the base
paper and by most click-fraud DL work.

Models
  DNNClassifier            - fully connected 128-64-32
  CNN1DClassifier          - Conv(32,k3) - Conv(64,k3) - GlobalMaxPool - Dense
  RNNClassifier            - SimpleRNN(64) over the feature sequence
  CNNAttentionClassifier   - PROPOSED EXTENSION: Conv stack + additive attention
                             pooling; exposes per-feature attention weights.
"""

from __future__ import annotations
import numpy as np
from autodiff import Tensor, Dense, Conv1D, SimpleRNN, AdditiveAttention, Adam


class _BaseNet:
    name = "net"

    def __init__(self, epochs=25, batch_size=256, lr=2e-3, seed=42,
                 class_weight=None, verbose=False):
        self.epochs, self.bs, self.lr = epochs, batch_size, lr
        self.seed, self.verbose = seed, verbose
        self.class_weight = class_weight
        self._built = False
        self.history_ = []

    # --- to be provided by subclasses ---
    def _build(self, d):
        raise NotImplementedError

    def _forward(self, X):
        raise NotImplementedError

    def _params(self):
        raise NotImplementedError

    # --- training loop ---
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if not self._built:
            self.rng = np.random.default_rng(self.seed)
            self._build(X.shape[1])
            self._built = True
        opt = Adam(self._params(), lr=self.lr)
        n = len(X)
        idx = np.arange(n)
        for ep in range(self.epochs):
            self.rng.shuffle(idx)
            tot = 0.0
            for s in range(0, n, self.bs):
                bi = idx[s:s + self.bs]
                xb, yb = X[bi], y[bi]
                logits = self._forward(xb)
                loss = logits.bce_with_logits(yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.data) * len(bi)
            self.history_.append(tot / n)
            if self.verbose:
                print(f"  [{self.name}] epoch {ep+1}/{self.epochs} loss={tot/n:.4f}")
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        outs = []
        for s in range(0, len(X), 4096):
            outs.append(self._forward(X[s:s + 4096]).data.reshape(-1))
        return np.concatenate(outs)

    def predict_proba(self, X):
        z = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class DNNClassifier(_BaseNet):
    name = "DNN"

    def _build(self, d):
        self.l1 = Dense(d, 128, self.rng)
        self.l2 = Dense(128, 64, self.rng)
        self.l3 = Dense(64, 32, self.rng)
        self.out = Dense(32, 1, self.rng)

    def _forward(self, X):
        h = Tensor(X)
        h = self.l1(h).relu()
        h = self.l2(h).relu()
        h = self.l3(h).relu()
        return self.out(h)

    def _params(self):
        return sum([l.params() for l in (self.l1, self.l2, self.l3, self.out)], [])


class CNN1DClassifier(_BaseNet):
    name = "CNN"

    def _build(self, d):
        self.c1 = Conv1D(1, 32, 3, self.rng)
        self.c2 = Conv1D(32, 64, 3, self.rng)
        self.fc = Dense(64, 64, self.rng)
        self.out = Dense(64, 1, self.rng)

    def _forward(self, X):
        h = Tensor(X.reshape(X.shape[0], X.shape[1], 1))
        h = self.c1(h).relu()
        h = self.c2(h).relu()
        h = h.max_pool_last()
        h = self.fc(h).relu()
        return self.out(h)

    def _params(self):
        return sum([l.params() for l in (self.c1, self.c2, self.fc, self.out)], [])


class RNNClassifier(_BaseNet):
    name = "RNN"

    def _build(self, d):
        self.proj = Dense(1, 16, self.rng)
        self.rnn = SimpleRNN(16, 64, self.rng)
        self.out = Dense(64, 1, self.rng)

    def _forward(self, X):
        B, D = X.shape
        h = Tensor(X.reshape(B * D, 1))
        h = self.proj(h).reshape(B, D, 16).tanh()
        h = self.rnn(h)
        return self.out(h)

    def _params(self):
        return self.proj.params() + self.rnn.params() + self.out.params()


class CNNAttentionClassifier(_BaseNet):
    """PROPOSED EXTENSION - 1D CNN with additive (Bahdanau) attention pooling.

    Instead of collapsing the convolutional feature map with a max-pool -- which
    throws away *where* the evidence came from -- the attention layer learns a
    soft weighting over feature positions. Two benefits:
      (a) accuracy: the model can emphasise the few behavioural signals that
          actually separate a bot from a human for THIS click;
      (b) interpretability: `attention_weights(X)` returns the per-position
          weights, which the dashboard turns into human-readable reason codes.
    """
    name = "CNN+Attention"

    def _build(self, d):
        self.c1 = Conv1D(1, 32, 3, self.rng)
        self.c2 = Conv1D(32, 64, 3, self.rng)
        self.att = AdditiveAttention(64, 32, self.rng)
        self.fc = Dense(64, 64, self.rng)
        self.out = Dense(64, 1, self.rng)
        self.k_total = 4          # total receptive-field offset of the 2 convs

    def _trunk(self, X):
        h = Tensor(X.reshape(X.shape[0], X.shape[1], 1))
        h = self.c1(h).relu()
        h = self.c2(h).relu()
        return h

    def _forward(self, X):
        h = self._trunk(X)
        ctx, self._alpha = self.att(h)
        g = self.fc(ctx).relu()
        return self.out(g)

    def attention_weights(self, X):
        """(N, L) attention distribution over convolutional positions."""
        X = np.asarray(X, dtype=np.float32)
        outs = []
        for s in range(0, len(X), 2048):
            h = self._trunk(X[s:s + 2048])
            _, a = self.att(h)
            outs.append(a.data.reshape(a.data.shape[0], -1))
        return np.concatenate(outs, axis=0)

    def feature_attention(self, X, n_features):
        """Map conv-position attention back onto the original feature axis."""
        A = self.attention_weights(X)                 # (N, L)
        L = A.shape[1]
        F = np.zeros((A.shape[0], n_features), dtype=np.float32)
        for j in range(L):
            for o in range(self.k_total + 1):
                if j + o < n_features:
                    F[:, j + o] += A[:, j]
        s = F.sum(axis=1, keepdims=True)
        return F / np.maximum(s, 1e-9)

    def _params(self):
        return sum([l.params() for l in (self.c1, self.c2, self.att,
                                         self.fc, self.out)], [])


class HumanBehaviourAutoencoder:
    """Density model of LEGITIMATE click behaviour (novelty head of the cascade).

    Trained only on human clicks, it learns the *joint* structure of a real
    session: that a desktop session emits no touch events, that a visitor who
    dwells for a minute and opens seven pages has also scrolled, that a browser
    reporting a rich user-agent also accepts cookies.

    A bot family that copies every marginal distribution individually but not
    the couplings between them therefore reconstructs badly, which is exactly
    the failure mode an axis-aligned Isolation Forest misses.
    """

    def __init__(self, hidden=(32, 12), epochs=40, batch_size=256, lr=3e-3,
                 seed=42, noise=0.08, verbose=False):
        self.h1, self.h2 = hidden
        self.epochs, self.bs, self.lr = epochs, batch_size, lr
        self.seed, self.noise, self.verbose = seed, noise, verbose

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        d = X.shape[1]
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        self.e1 = Dense(d, self.h1, rng); self.e2 = Dense(self.h1, self.h2, rng)
        self.d1 = Dense(self.h2, self.h1, rng); self.d2 = Dense(self.h1, d, rng)
        params = sum([l.params() for l in (self.e1, self.e2, self.d1, self.d2)], [])
        opt = Adam(params, lr=self.lr)
        idx = np.arange(len(X))
        for ep in range(self.epochs):
            rng.shuffle(idx)
            tot = 0.0
            for s in range(0, len(X), self.bs):
                xb = X[idx[s:s + self.bs]]
                # denoising objective: forces the model to use cross-feature
                # structure instead of learning the identity map
                xn = xb + self.noise * rng.normal(size=xb.shape).astype(np.float32)
                rec = self._decode(Tensor(xn))
                diff = rec - Tensor(xb)
                loss = (diff * diff).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.data) * len(xb)
            if self.verbose:
                print(f"  [HBAE] epoch {ep+1}/{self.epochs} mse={tot/len(X):.5f}")
        return self

    def _decode(self, t):
        h = self.e1(t).relu()
        h = self.e2(h).tanh()
        h = self.d1(h).relu()
        return self.d2(h)

    def reconstruction_error(self, X):
        X = np.asarray(X, dtype=np.float32)
        errs = []
        for s in range(0, len(X), 4096):
            xb = X[s:s + 4096]
            rec = self._decode(Tensor(xb)).data
            errs.append(((rec - xb) ** 2).mean(axis=1))
        return np.concatenate(errs)
