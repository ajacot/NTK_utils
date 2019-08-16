import torch
import numpy

def forward_grad(y, x, dx, retain_graph=False, create_graph=False):
	var_dy = torch.zeros_like(y, requires_grad=True)
	ddx = torch.autograd.grad(y, x, var_dy, allow_unused=True, create_graph=True)
	return torch.autograd.grad(ddx, var_dy, dx, allow_unused=True, retain_graph=retain_graph, create_graph=create_graph)

def scalar_product(x, y):
	return sum([(xx*yy).sum() for (xx, yy) in zip(x, y)])

def power_method(T, Ds, num_steps=51, mult=1.0):
	K = len(Ds)
	norm_Ds = [1.0 for k in range(K)]
	for t in range(num_steps):
		for k in range(K):
			Dk = T(Ds[k])
			for j in range(k):
				sc = scalar_product(Dk, Ds[j])
				Dk = [dk - sc*dj for dk, dj in zip(Dk, Ds[j])]
			norm_Ds[k] = torch.sqrt(scalar_product(Dk, Dk)) * torch.sign((Dk[0]*Ds[k][0]).sum())
			Ds[k] = [d / norm_Ds[k] for d in Dk]
	return norm_Ds, Ds
