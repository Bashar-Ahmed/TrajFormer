import torch

def sinkhorn(x, max_iter=2, epsilon=1e-8):
    
    factor = x.shape[-2]/x.shape[-1]
    k = x.softmax(dim=-1)
    for i in range(max_iter):
        col_sum = k.sum(dim=-2, keepdim=True)
        col_sum[col_sum<=epsilon] = 1.
        k = k.div(col_sum)
        row_sum = k.sum(dim=-1, keepdim=True)
        k = k.div(row_sum)
    return k
