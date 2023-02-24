import torch

def sinkhorn(x, max_iter=3, epsilon=1e-7):
    
    k = torch.softmax(x, dim=-1)
    for i in range(max_iter):
        col_sum = torch.sum(k, -2, keepdim=True) + epsilon
        k = k / col_sum
        row_sum = torch.sum(k, -1, keepdim=True) + epsilon
        k = k / row_sum
    return k
