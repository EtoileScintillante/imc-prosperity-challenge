import numpy as np
import itertools
import math

rates = np.array([[1, 1.34, 1.98, 0.64],
                  [0.72, 1, 1.45, 0.52],
                  [0.48, 0.7, 1, 0.31],
                  [1.49, 1.95, 3.1, 1]
                 ])
products = {0:'Shell', 1:'Snowball', 2:'Pizza', 3:'Nuggets'}



def amount(seq):
    if not seq:
        return 1
    prod = rates[0, seq[0]] * rates[seq[-1], 0]
    L = len(seq)
    for i in range(L - 1):
        prod *= rates[seq[i], seq[i + 1]]
    return prod


def maximize(L):
    seqs = itertools.product(*[range(0, 4) for _ in range(L)])
    max_val = float('-inf')
    argmax = []
    for seq in seqs:
        p = amount(seq)
        if math.isclose(p, max_val):
            argmax.append(seq)
        elif p > max_val:
            max_val = p
            argmax = [seq]
    return (argmax, max_val)

for L in range(0,5):
    print(maximize(L))

argmax, _ = maximize(4)
print("Optimal sequences of trades:")
for seq in argmax:
    res = ' -> '.join([products[0]] + [products[i] for i in seq] + [products[0]])
    print(res)
