from pathlib import Path
import torch,numpy as np,json,hashlib,math
ROOT=Path(__file__).resolve().parent
CACHE=ROOT.parent/'coco/coco_data/coco_val2017_preprocessed_32.npz'
SPLIT=ROOT.parent/'coco/prepared/split.json'
def save_json(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False));tmp.replace(p)
def patchify(x):return x.reshape(-1,3,4,8,4,8).permute(0,2,4,1,3,5).reshape(-1,192)
def prepare():
 torch.set_num_threads(2);out=ROOT/'data';out.mkdir(exist_ok=True)
 if (out/'DATA_READY.json').exists():return
 with np.load(CACHE,allow_pickle=False) as z:
  raw=torch.from_numpy(z['images'].copy()).permute(0,3,1,2).float()/127.5-1
  names=z['names'].tolist()
 split=json.loads(SPLIT.read_text());lookup={n:i for i,n in enumerate(names)}
 train_ids=[lookup[n] for n in split['train']];held_ids=[lookup[n] for n in split['heldout']]
 assert len(train_ids)==4000 and len(held_ids)==1000 and not set(train_ids)&set(held_ids)
 train_real=raw[train_ids];held_real=raw[held_ids]
 g=torch.Generator().manual_seed(20261024)
 coeff=torch.randn(5000,3,32,32,generator=g)
 u=torch.arange(32).float();freq2=1+u[:,None]**2+u[None,:]**2
 c=torch.cos(torch.pi*(2*torch.arange(32)[None,:]+1)*torch.arange(32)[:,None]/64)
 c[0]*=math.sqrt(1/32);c[1:]*=math.sqrt(2/32)
 metadata={'train_count':4000,'validation_count':512,'test_count':488,'resolution':32,'patch':8,'pca_dimension':192,
  'normalization':'subtract train-only per-location/channel mean; divide train-only scalar RMS; no per-image scaling or clipping',
  'noise':'independent N(0,I); z=0.5*x+0.5*epsilon; expected equal image/noise energy',
  'synthetic_generator':'same independent Gaussian DCT coefficients for all alpha, multiply std=(1+u^2+v^2)^(-alpha/4), orthonormal inverse DCT; all alpha finite with full Gaussian support',
  'cache_sha256':hashlib.sha256(CACHE.read_bytes()).hexdigest(),'original_split_sha256':hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
  'split':{'train':split['train'],'validation':split['heldout'][:512],'test':split['heldout'][512:]},'datasets':{}}
 for name,alpha in [('gaussian_a0',0),('gaussian_a1',1),('gaussian_a2',2),('gaussian_a3',3),('real_coco',None)]:
  if alpha is None:tr,ho=train_real,held_real
  else:
   # Same random draws, orthonormal basis and sample counts; only the spectrum exponent changes.
   x=torch.einsum('uh,bcuv,vw->bchw',c,coeff*freq2.pow(-alpha/4)[None,None],c)
   tr,ho=x[:4000],x[4000:]
  mean=tr.mean(0,keepdim=True);scale=(tr-mean).square().mean().sqrt()
  tr=(tr-mean)/scale;ho=(ho-mean)/scale
  assert abs(tr.square().mean().item()-1)<1e-5
  pt=patchify(tr).double();mu=pt.mean(0);center=pt-mu;cov=center.T@center/len(pt)
  lam,U=torch.linalg.eigh(cov);lam=lam.flip(0).clamp_min(0);U=U.flip(1)
  valp=patchify(ho[:512]).double()-mu;en=(valp@U).square().mean(0)
  path=out/f'{name}.pt';torch.save({'train':tr.contiguous(),'validation':ho[:512].contiguous(),'test':ho[512:].contiguous(),
   'mean':mean,'scale':scale,'pca_mean':mu,'eigenvalues':lam,'eigenvectors':U,'validation_energy_per_pc':en},path)
  d={'alpha':alpha,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'train_energy':tr.square().mean().item(),
    'validation_energy':ho[:512].square().mean().item(),'test_energy':ho[512:].square().mean().item(),
    'eigenvalues':lam.tolist(),'train_cumulative_energy':(lam.cumsum(0)/lam.sum()).tolist(),
    'validation_cumulative_energy':(en.cumsum(0)/en.sum()).tolist(),'effective_rank':float(lam.sum().square()/lam.square().sum()),
    'E16':float(lam[:16].sum()/lam.sum()),'E50':float(lam[:50].sum()/lam.sum()),'normalization_rms':float(scale)}
  metadata['datasets'][name]=d;print(name,'E16',round(d['E16'],4),'E50',round(d['E50'],4),'effective rank',round(d['effective_rank'],2),flush=True)
 save_json(out/'DATA_READY.json',metadata)
if __name__=='__main__':prepare()
