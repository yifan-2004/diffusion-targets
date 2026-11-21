"""Paired, checkpointed ViT experiment. No report edits and no external/cloud actions."""
import os,time,json,csv,hashlib,signal,fcntl,traceback,copy,platform,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from model import ViT
from prepare import prepare,save_json
ROOT=Path(__file__).resolve().parent
STOP=False

def signal_stop(*_):
 global STOP;STOP=True
signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)

@torch.no_grad()
def evaluate(models,x,noise,device):
 out={h:[] for h in models}
 for m in models.values():m.eval()
 for start in range(0,len(x),64):
  xx=x[start:start+64].to(device);ee=noise[start:start+64].to(device);z=.5*(xx+ee)
  for h,m in models.items():
   target=xx if h=='x' else ee
   out[h].extend((m(z)-target).square().mean((1,2,3)).cpu().tolist())
 for m in models.values():m.train()
 return out

def plateau(curve,head):
 # Fixed-noise validation, median of two consecutive 500-update windows. A diagnostic, not proof of convergence.
 rows=[r for r in curve if r['step']>0]
 if len(rows)<10:return {'plateau':False,'relative_change':None}
 earlier=np.median([r[head] for r in rows[-10:-5]]);later=np.median([r[head] for r in rows[-5:]])
 change=float((earlier-later)/max(earlier,1e-12))
 return {'plateau':abs(change)<.01,'relative_change':change}

def checkpoint(path,models,opts,g,step,curve,elapsed):
 tmp=path.with_suffix('.tmp')
 torch.save({'step':step,'models':{h:{k:v.detach().cpu() for k,v in m.state_dict().items()} for h,m in models.items()},
  'optimizers':{h:o.state_dict() for h,o in opts.items()},'generator':g.get_state(),'curve':curve,'elapsed':elapsed},tmp);tmp.replace(path)

def main():
 lock=open(ROOT/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
 protocol={'datasets':['gaussian_a0','gaussian_a1','gaussian_a2','gaussian_a3','real_coco'],
  'seeds':[0,1,2],'heads':['x','eps'],'t':.5,'resolution':32,'patch':8,'bottleneck':16,'width':96,'depth':2,'heads_attention':3,
  'model':'plain ViT: linear 192->16->96 patch embedding; fixed 2D sin-cos position; LayerNorm; global attention; GELU MLP ratio4; linear RGB head; zero output init; no dropout; all parameters trained',
  'batch':32,'lr':.0002,'betas':[.9,.95],'weight_decay':0,'gradient_clip':1,'precision':'FP32',
  'primary_equal_budget':3000,'max_steps':6000,'validation_every':100,'checkpoint_every':500,
  'extension':'after 3000 updates, if either paired head is not plateaued, extend BOTH in blocks of500 up to6000; primary comparison remains equal budget3000; never select a checkpoint by test error',
  'plateau_criterion':'absolute relative change between validation medians of previous500 and latest500updates below1% for BOTH heads; min3000; operational stability only',
  'speed_thresholds':[.95,.90,.75,.50,.25,.10],
  'speed_rule':'first validation checkpoint below fixed MSE threshold and next two checks also below; not reached is null; compare x/eps within same dataset',
  'curves':'per-update fresh-minibatch MSE; every100 fixed-noise train probe256 and validation512; final test488 independent from validation; direct target MSE, common velocity MSE exactly4x',
  'device':device,'torch_version':torch.__version__,'machine':platform.machine(),'cpu_count':os.cpu_count(),
  'sources':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['model.py','prepare.py','run.py']}}
 proto=ROOT/'protocol.json'
 if proto.exists():
  prior=json.loads(proto.read_text());assert prior==protocol,'Protocol/code changed: do not mix runs.'
 else:save_json(proto,protocol)
 prepare();results=ROOT/'runs';results.mkdir(exist_ok=True);tick=time.perf_counter();completed=0
 for data_name in protocol['datasets']:
  data=torch.load(ROOT/'data'/f'{data_name}.pt',weights_only=True)
  tr,va,test=data['train'],data['validation'],data['test']
  del data
  for seed in protocol['seeds']:
   name=f'{data_name}_s{seed}';folder=results/name;folder.mkdir(exist_ok=True);done=folder/'result.json'
   if done.exists():completed+=1;continue
   torch.manual_seed(8100+seed);m=ViT().to(device);models={'x':m,'eps':copy.deepcopy(m)}
   assert all(torch.equal(models['x'].state_dict()[k],v) for k,v in models['eps'].state_dict().items())
   opts={h:torch.optim.AdamW(m.parameters(),lr=protocol['lr'],betas=(.9,.95),weight_decay=0) for h,m in models.items()}
   g=torch.Generator().manual_seed(51000+seed)
   valnoise=torch.randn(va.shape,generator=torch.Generator().manual_seed(42050))
   trainnoise=torch.randn(tr[:256].shape,generator=torch.Generator().manual_seed(42051))
   curve=[];step=0;prior_elapsed=0;ckpath=folder/'checkpoint.pt';pairtick=time.perf_counter()
   if ckpath.exists():
    ck=torch.load(ckpath,map_location='cpu',weights_only=True)
    for h in models:models[h].load_state_dict(ck['models'][h]);opts[h].load_state_dict(ck['optimizers'][h])
    g.set_state(ck['generator']);step=ck['step'];curve=ck['curve'];prior_elapsed=ck['elapsed'];del ck
   csvpath=folder/'training.csv'
   if step and csvpath.exists():
    rows=list(csv.DictReader(csvpath.open()));rows=[r for r in rows if int(r['step'])<=step]
    with csvpath.open('w',newline='') as f:
     w=csv.DictWriter(f,fieldnames=['step','x','eps','seconds']);w.writeheader();w.writerows(rows)
   log=csvpath.open('a' if step else 'w',newline='');w=csv.writer(log)
   if step==0:w.writerow(['step','x','eps','seconds'])
   initial_step=step
   while True:
    if step%100==0 and not any(r['step']==step for r in curve):
     ev=evaluate(models,va,valnoise,device);tv=evaluate(models,tr[:256],trainnoise,device)
     row={'step':step,'x':float(np.mean(ev['x'])),'eps':float(np.mean(ev['eps'])),
      'train_probe_x':float(np.mean(tv['x'])),'train_probe_eps':float(np.mean(tv['eps'])),
      'seconds':prior_elapsed+time.perf_counter()-pairtick};curve.append(row)
     save_json(folder/'curve.json',curve)
     state={'state':'training','pair':name,'step':step,'completed_pairs':completed,'total_pairs':15,'x_validation':row['x'],'eps_validation':row['eps'],
      'elapsed_seconds_current_invocation':time.perf_counter()-tick,'pid':os.getpid(),'device':device}
     save_json(ROOT/'status.json',state);print(name,step,'x',round(row['x'],5),'eps',round(row['eps'],5),flush=True)
    if step%500==0 and step!=initial_step:
     log.flush();checkpoint(ckpath,models,opts,g,step,curve,prior_elapsed+time.perf_counter()-pairtick)
    if STOP:
     log.flush();checkpoint(ckpath,models,opts,g,step,curve,prior_elapsed+time.perf_counter()-pairtick)
     save_json(ROOT/'status.json',{'state':'stopped_saved','pair':name,'step':step});return
    if step>=3000 and step%500==0:
     statuses={h:plateau(curve,h) for h in models}
     if all(v['plateau'] for v in statuses.values()) or step==6000:break
    ix=torch.randint(len(tr),(32,),generator=g);xx=tr[ix].to(device);ee=torch.randn(xx.shape,generator=g).to(device);z=.5*(xx+ee);losses={}
    for h,m in models.items():
     target=xx if h=='x' else ee;opt=opts[h];opt.zero_grad(set_to_none=True)
     loss=(m(z)-target).square().mean()
     if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {name} {h} {step}')
     loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses[h]=loss.item()
    step+=1;w.writerow([step,losses['x'],losses['eps'],prior_elapsed+time.perf_counter()-pairtick])
   log.close();testnoise=torch.randn(test.shape,generator=torch.Generator().manual_seed(42052));testev=evaluate(models,test,testnoise,device)
   final={'dataset':data_name,'seed':seed,'step':step,'curve':curve,'plateau':{h:plateau(curve,h) for h in models},
    'primary_3000':next(r for r in curve if r['step']==3000),
    'test_mse':{h:float(np.mean(testev[h])) for h in models},'test_by_image':testev,
    'parameters_per_model':sum(p.numel() for p in models['x'].parameters()),'seconds':prior_elapsed+time.perf_counter()-pairtick}
   final['checkpoints']={}
   for h,m in models.items():
    path=folder/f'{h}_final.pt';torch.save({'model':{k:v.cpu() for k,v in m.state_dict().items()},'optimizer':opts[h].state_dict(),'step':step,'head':h,'dataset':data_name,'seed':seed},path)
    final['checkpoints'][h]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
   save_json(done,final);ckpath.unlink(missing_ok=True);completed+=1
   print('PAIR COMPLETE',name,step,final['test_mse'],flush=True);del models,opts
  del tr,va,test
 save_json(ROOT/'status.json',{'state':'training_complete_analyzing','completed_pairs':15,'total_pairs':15,'elapsed_seconds':time.perf_counter()-tick})
 subprocess.run([sys.executable,str(ROOT/'analyze.py')],cwd=ROOT,check=True)
 save_json(ROOT/'COMPLETE.json',{'completed_pairs':15,'models':30,'elapsed_seconds':time.perf_counter()-tick,'finished_unix':time.time()})
 save_json(ROOT/'status.json',{'state':'complete','completed_pairs':15,'total_pairs':15,'elapsed_seconds':time.perf_counter()-tick})
if __name__=='__main__':
 try:main()
 except Exception:
  save_json(ROOT/'FAILURE.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
