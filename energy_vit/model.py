"""Plain ViT with explicit linear patch bottleneck; no convolutional mixing."""
import torch
from torch import nn
from torch.nn import functional as F
class Attention(nn.Module):
 def __init__(self,width=96,heads=3):
  super().__init__();self.heads=heads;self.qkv=nn.Linear(width,width*3);self.out=nn.Linear(width,width)
 def forward(self,x):
  b,n,d=x.shape;q,k,v=self.qkv(x).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
  a=F.scaled_dot_product_attention(q,k,v)
  return self.out(a.transpose(1,2).reshape(b,n,d))
class Block(nn.Module):
 def __init__(self,width,heads):
  super().__init__();self.n1=nn.LayerNorm(width);self.n2=nn.LayerNorm(width);self.attn=Attention(width,heads)
  self.mlp=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
 def forward(self,x):
  x=x+self.attn(self.n1(x));return x+self.mlp(self.n2(x))
class ViT(nn.Module):
 def __init__(self,width=96,depth=2,heads=3,bottleneck=16):
  super().__init__();self.patch1=nn.Linear(192,bottleneck,bias=False);self.patch2=nn.Linear(bottleneck,width)
  self.blocks=nn.Sequential(*[Block(width,heads) for _ in range(depth)]);self.norm=nn.LayerNorm(width);self.output=nn.Linear(width,192)
  y,x=torch.meshgrid(torch.arange(4),torch.arange(4),indexing='ij');freq=1/10000**(torch.arange(width//4)/(width//4))
  self.register_buffer('pos',torch.cat([(y.flatten()[:,None]*freq).sin(),(y.flatten()[:,None]*freq).cos(),(x.flatten()[:,None]*freq).sin(),(x.flatten()[:,None]*freq).cos()],-1)[None])
  nn.init.orthogonal_(self.patch1.weight);nn.init.orthogonal_(self.patch2.weight);nn.init.zeros_(self.patch2.bias)
  nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def forward(self,x):
  b=len(x);patches=x.reshape(b,3,4,8,4,8).permute(0,2,4,1,3,5).reshape(b,16,192)
  h=self.patch2(self.patch1(patches))+self.pos
  out=self.output(self.norm(self.blocks(h)))
  return out.reshape(b,4,4,3,8,8).permute(0,3,1,4,2,5).reshape(b,3,32,32)
