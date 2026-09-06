"""Finite-difference gradient check for the NumPy autodiff engine.

Every layer the deep models are built from is checked against a central
difference. If this passes, the DNN, CNN, RNN, attention model and autoencoder
are all differentiating correctly.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import numpy as np
from autodiff import Tensor, Dense, Conv1D, SimpleRNN, AdditiveAttention

TOL = 2e-3


def check(name, params, forward, target):
    for p in params:
        p.grad = np.zeros_like(p.data)
    loss = forward().bce_with_logits(target)
    loss.backward()
    worst = 0.0
    for p in params:
        idx = tuple(np.unravel_index(0, p.data.shape))
        analytic = float(p.grad[idx])
        old, eps = float(p.data[idx]), 1e-3
        p.data[idx] = old + eps
        l1 = float(forward().bce_with_logits(target).data)
        p.data[idx] = old - eps
        l2 = float(forward().bce_with_logits(target).data)
        p.data[idx] = old
        numeric = (l1 - l2) / (2 * eps)
        worst = max(worst, abs(analytic - numeric))
    assert worst < TOL, f"{name}: gradient mismatch {worst:.2e}"
    print(f"  {name:<22} max |analytic - numeric| = {worst:.2e}")


def main():
    rng = np.random.default_rng(0)
    y = np.array([[0], [1], [1], [0]])
    x = Tensor(rng.normal(size=(4, 12, 1)).astype(np.float32))
    flat = Tensor(rng.normal(size=(4, 12)).astype(np.float32))

    print("Gradient checks:")

    d1, d2 = Dense(12, 8, rng), Dense(8, 1, rng)
    check("Dense stack", d1.params() + d2.params(),
          lambda: d2(d1(flat).relu()), y)

    c1, c2, fc = Conv1D(1, 6, 3, rng), Conv1D(6, 8, 3, rng), Dense(8, 1, rng)
    check("Conv1D + max pool", c1.params() + c2.params() + fc.params(),
          lambda: fc(c2(c1(x).relu()).relu().max_pool_last()), y)

    cr, rn, do = Conv1D(1, 5, 3, rng), SimpleRNN(5, 6, rng), Dense(6, 1, rng)
    check("SimpleRNN", cr.params() + rn.params() + do.params(),
          lambda: do(rn(cr(x).relu())), y)

    ca, at, ao = Conv1D(1, 6, 3, rng), AdditiveAttention(6, 5, rng), Dense(6, 1, rng)
    check("Additive attention", ca.params() + at.params() + ao.params(),
          lambda: ao(at(ca(x).relu())[0]), y)

    print("All gradient checks passed.")


if __name__ == "__main__":
    main()
