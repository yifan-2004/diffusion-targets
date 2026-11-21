"""One independent, protocol-matched PCA training arm or throughput benchmark."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import width_model as base

def save(path, value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def patches(x):
 return x.unfold(2,8,8).unfold(3,8,8).permute(0,2,3,1,4,5).reshape(-1,192)

@torch.no_grad()
def alignment(m,U,lam,heldout_second):
 w1=m.patch1.weight.detach().cpu().reshape(m.patch1.out_channels,192).double()
 w2=m.patch2.weight.detach().cpu().reshape(192,-1).double()
 _,sing,Vh=torch.linalg.svd(w2@w1,full_matrices=False)
 rank=int((sing>sing[0]*1e-6).sum());Q=Vh[:rank].T
 gamma=(Q.T@U).square().sum(0)
 return dict(rank=rank,principal_direction_retention=gamma.tolist(),
  image_energy_train=float((gamma*lam).sum()/lam.sum()),
  image_energy_heldout=float(torch.trace(Q.T@heldout_second@Q)/torch.trace(heldout_second)),
  white_noise_energy_expected=rank/192,
  top12_overlap=float(gamma[:12].mean()),bottom12_overlap=float(gamma[-12:].mean()))


ROOT=Path(__file__).resolve().parent
PRIOR=ROOT/'coco_pca_observational_runs/20260926_025644'

@torch.no_grad()
def evaluate(m,val,head,U=None):
    m.eval();noise=torch.randn(val.shape,generator=torch.Generator().manual_seed(42050))
    by=[];errors=torch.zeros(192,dtype=torch.float64)
    for start in range(0,len(val),64):
        x=val[start:start+64].to(base.DEVICE);e=noise[start:start+64].to(base.DEVICE);z=.5*(x+e)
        pred=m(z,torch.full((len(x),),.5,device=base.DEVICE))
        vp=2*(pred-z) if head=='x' else (2*(z-pred) if head=='eps' else pred)
        err=(vp-x+e).cpu();by.extend(err.square().mean((1,2,3)).tolist())
        if U is not None:errors+=(patches(err)@U.float()).double().square().sum(0)
    m.train()
    if U is None:return float(np.mean(by))
    pc=errors/(len(val)*16);assert abs(float(pc.mean())-np.mean(by))<1e-5
    return dict(by_image=by,pc_v_mse=pc.tolist(),band_contributions=[float(pc[:12].sum()/192),float(pc[12:48].sum()/192),float(pc[48:].sum()/192)])

def main(args):
    torch.set_num_threads(2)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    name=f'natural_b{args.bottleneck}_{args.head}_s{args.seed}'
    if not args.benchmark and (out/f'{name}.json').exists():return
    data=torch.load(args.data,weights_only=True)
    tr=data['train'].to(base.DEVICE);va=data['heldout'];pca=data['pca']
    del data
    m=base.model(args.bottleneck,args.seed)
    opt=torch.optim.AdamW(m.parameters(),lr=.0002,betas=(.9,.95),weight_decay=0)
    gen=torch.Generator().manual_seed(51000+args.seed)
    record=dict(bottleneck=args.bottleneck,head=args.head,seed=args.seed,parameters=sum(p.numel() for p in m.parameters()),curve=[])
    steps=args.benchmark or 1500;first=0;ckpath=out/f'{name}_latest.pt'
    if not args.benchmark and ckpath.exists():
        ck=torch.load(ckpath,map_location='cpu',weights_only=True)
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);gen.set_state(ck['generator']);record=ck['record'];first=ck['step']
    if args.gate:
        (out/f'ready_{args.worker}.json').write_text('{}')
        while not args.gate.exists():time.sleep(.05)
    tick=time.perf_counter()
    for step in range(first,steps+1):
        if not args.benchmark and step in [0,100,250,500,750,1000,1250,1500] and not any(q['step']==step for q in record['curve']):
            mse=evaluate(m,va[:256],args.head)
            diag=alignment(m,pca['eigenvectors'],pca['eigenvalues'],pca['heldout_second_moment'])
            assert np.isfinite(mse)
            record['curve'].append(dict(step=step,v_mse=mse,**diag))
            save(out/f'{name}_progress.json',record)
            print(f'{name} {step}/{steps} mse={mse:.6f}',flush=True)
            if step in [500,1000]:
                tmp=ckpath.with_suffix('.tmp');torch.save(dict(step=step,model=m.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),record=record),tmp);tmp.replace(ckpath)
        if step==steps:break
        x=tr[torch.randint(len(tr),(32,),generator=gen)]
        t=(.1+.8*torch.rand(32,generator=gen)).to(base.DEVICE)
        eps=torch.randn(x.shape,generator=gen).to(base.DEVICE);tt=t[:,None,None,None]
        z=tt*x+(1-tt)*eps;pred=m(z,t)
        vp=(pred-z)/(1-tt) if args.head=='x' else ((z-pred)/tt if args.head=='eps' else pred)
        loss=(vp-x+eps).square().mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    if args.benchmark:
        if base.DEVICE=='mps':torch.mps.synchronize()
        save(out/f'done_{args.worker}.json',dict(steps=steps,seconds=time.perf_counter()-tick));return
    record['final']=evaluate(m,va,args.head,pca['eigenvectors']);record['seconds']=time.perf_counter()-tick
    tmp=(out/f'{name}.pt').with_suffix('.tmp');torch.save(m.cpu().state_dict(),tmp);tmp.replace(out/f'{name}.pt')
    save(out/f'{name}.json',record)
    ckpath.unlink(missing_ok=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--bottleneck',type=int,required=True);ap.add_argument('--head',choices=['x','eps','v'],required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--benchmark',type=int,default=0);ap.add_argument('--gate',type=Path);ap.add_argument('--worker',type=int,default=0)
    main(ap.parse_args())
