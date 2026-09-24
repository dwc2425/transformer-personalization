# Week 1 — Self-Attention from Scratch

**Date:** 2026-09-24

## Softmax numerical stability
Ran softmax on entries around 1000 without subtracting the row max first:
`exp(1000)` overflows to `inf`, and `inf / inf` becomes `nan`. Output was
`[[nan nan nan]]`. Fixed by subtracting the row max before exponentiating
(this doesn't change the softmax result, since it just rescales the ratio
inside the exponent). After the fix, same input gives a normal distribution:
`[[0.245 0.665 0.090]]`.

## Task 4 — effect of scaling Q, K
With X = I, n=6, d_kq=2, scaling Q and K by c before computing attention
weights:
- c = 0.1: rows nearly uniform (~0.167 each, since 1/6 ≈ 0.167)
- c = 1: rows noticeably non-uniform but still soft
- c = 10: rows collapse to near one-hot (one entry ≈ 1, rest ≈ 0)

Scaling up the pre-softmax scores pushes softmax toward the max entry
(sharper distribution); scaling down flattens it toward uniform. This is
why scaled attention divides by sqrt(head_size) — to keep the dot products
in a reasonable range at initialization instead of saturating softmax.
