import torch

def sinkhorn(x, dim=-1, max_iter=2, epsilon=1e-12):

    k = x.softmax(dim=dim)
    for i in range(max_iter):
        col_sum = k.sum(dim=dim-1, keepdim=True)
        col_sum[col_sum<=epsilon] = 1.
        k = k.div(col_sum)
        row_sum = k.sum(dim=dim, keepdim=True)
        k = k.div(row_sum)
    return k
