import numpy as np
from src.attention import attention_weights
from src.simple_transformer import simple_transformer, selection_matrix
 
Q = np.array([[np.log(2)], [0.0], [1.0]])
K = np.array([[1.0], [0.0], [-1.0]])
V = np.array([[2.0], [8.0], [20.0]])
u = np.array([2.0])
f = lambda t: max(0.0, t - 3)
 
 
def show(tag, B, A, T, rows):
    for i in rows:
        print(tag, f"row {i+1}: weights={np.round(B[i],4)} A={A[i,0]:.4f} T={T[i]:.4f}")
 
 
for S in ([0, 1], [0, 1, 2]):
    B, A, T = simple_transformer(Q, K, V, u, S, fs=f)
    show(f"masked S={[s+1 for s in S]}", B, A, T, [0, 2])
B, A, T = simple_transformer(Q, K, 2 * V, u, [0, 1], fs=f)
show("Q4 2V, S={1,2}", B, A, T, [0])
 
# Week 1 unmasked function, with X_S as the input and Q, K passed directly.
# Zeroed rows/cols give logit 0, so e^0 = 1 stays in the softmax denominator
# for every unselected key. They are not excluded, so the selected items'
# weights get diluted (row 3 becomes uniform over all 3, not just S).
X, _ = selection_matrix(3, [0, 1])
Bu = attention_weights(X, Q, K)
Au = Bu @ X @ V
show("unmasked S={1,2}", Bu, Au, np.array([f(z) for z in Au @ u]), [2])
