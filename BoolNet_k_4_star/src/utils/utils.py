import torch
import torch.nn.functional as F

def CrossEntropy2D(outputs, labels):
    """
    outputs: 4D Tensor N C H W
    labels:  3D Tensor N H W
    """
    N, H, W = labels
    
    if outputs.size()[2] != H or outputs.size()[3] != W:
      outputs = F.interpolate(outputs, size = (H, W), mode = "bilinear")
    
    outputs = outputs.transpose(1, 2).transpose(2, 3).reshape(-1, C)
    
    labels = labels.reshape(-1)
    
    return F.cross_entropy(outputs, labels)

def automatic_gradient_clip(network, ratio = 0.05, eps = 1e-5):
    for name, parameter in network.named_parameters():
      
      if parameter.grad is not None and "BConv" in name:
        
        weight = abs(parameter.data)
        
        w_sign = parameter.data.sign()
        
        grad = abs(parameter.grad)
        
        g_sign = parameter.grad.sign()
                
        scalar = grad/weight.clamp(min = eps)
        
        scalar = torch.where((scalar > ratio), (ratio/scalar), torch.ones_like(scalar))
        
        #scalar = torch.where(w_sign == g_sign, scalar, torch.ones_like(scalar))
        
        parameter.grad.mul_(scalar)
          
        
        
        
        
    
      