import torch
def trans_phase(a,b):
    assert a.shape[-1] == 2 and b.shape[-1] == 2
    d1 = torch.sum(a*b)
    temp = a*torch.flip(b,[0,2])
    d2_v = temp[:,:,0] - temp[:,:,1]
    d2 = torch.sum(d2_v)
    intt = torch.sqrt(d1**2 + d2**2)
    return d1/intt, d2/intt

def trans_tensor(a,b):
    c,s = trans_phase(a,b)
    return torch.stack([c*a[:,:,0]-s*a[:,:,1], s*a[:,:,0]+c*a[:,:,1]],-1)

def dist_real(a,b):
    return min(torch.norm(a-b), torch.norm(a+b))/torch.norm(b)

def dist_comp(a,b):
    a_m = trans_tensor(a,b)
    return torch.norm(a_m-b)/torch.norm(b)
