import os
import clip
import torch
a = [[1,0,2,1],[2,0,1,1]]
b = [[1,0],[2,1],[1,1],[0,1]],[[0,1],[0,1],[1,1],[2,2]]
a = torch.Tensor(a)
b = torch.Tensor(b)
print(torch.einsum('ij,ijk->ik',a,b))
