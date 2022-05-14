import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def multi_ssm(model,
              x0: torch.Tensor,
              t: torch.LongTensor,
              e: torch.Tensor,
              b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    
    x.requires_grad_(True)
    
    vectors = torch.randn_like(x)
    grad1 = model(x, t.float())
    gradv = torch.sum(grad1 * vectors)
    grad2 = torch.autograd.grad(gradv, x, create_graph=True)[0]
    
    loss1 = torch.sum(grad1 * grad1, dim=[-1,-2,-3]) / 2.
    loss2 = torch.sum(vectors * grad2, dim=[-1,-2,-3])
    
     if keepdim:
        return loss1 + loss2
    else:
        return (loss1+loss2).mean(dim=0)   
    
   
loss_registry = {
    'simple': noise_estimation_loss,
    'multi_ssm': multi_ssm,
}
