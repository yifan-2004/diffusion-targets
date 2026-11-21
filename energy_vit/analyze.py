"""Analyze measured curves only; never interpolate untrained tails."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
ROOT=Path(__file__).resolve().parent
from prepare import save_json
LABELS={'gaussian_a0':'Gaussian alpha=0','gaussian_a1':'Gaussian alpha=1','gaussian_a2':'Gaussian alpha=2','gaussian_a3':'Gaussian alpha=3','real_coco':'Real COCO'}
COLORS={'x':'#2563eb','eps':'#e87520'}

def stable_cross(curve,head,threshold):
 for i in range(len(curve)-2):
  if all(r[head]<=threshold for r in curve[i:i+3]):return curve[i]['step']
 return None

def main():
 torch.set_num_threads(2)
 protocol=json.loads((ROOT/'protocol.json').read_text());meta=json.loads((ROOT/'data/DATA_READY.json').read_text())
 records=[json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('*/result.json'))]
 assert len(records)==15
 for r in records:
  assert len(r['curve'])==1+r['step']//100 and r['step']>=3000
  assert all(np.isfinite([row[h] for row in r['curve']]).all() for h in ['x','eps'])
  r['encoder_retention']={}
  pdata=torch.load(ROOT/'data'/f'{r["dataset"]}.pt',weights_only=True)
  U=pdata['eigenvectors'];lam=pdata['eigenvalues']
  del pdata
  for h,item in r['checkpoints'].items():
   path=Path(item['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
   ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['step']==r['step'] and ck['head']==h
   weight=ck['model']['patch1.weight'].double();_,sv,Vh=torch.linalg.svd(weight,full_matrices=False)
   rank=int((sv>sv[0]*1e-6).sum());Q=Vh[:rank].T;ret=(Q.T@U).square().sum(0)
   r['encoder_retention'][h]={'rank':rank,'train_image_energy':float((ret*lam).sum()/lam.sum()),'noise_energy_expected':rank/192}
 summary=[]
 for d in protocol['datasets']:
  rr=[r for r in records if r['dataset']==d];assert len(rr)==3
  for h in ['x','eps']:
   val=np.array([r['primary_3000'][h] for r in rr]);test=np.array([r['test_mse'][h] for r in rr]);thresholds={}
   for th in protocol['speed_thresholds']:
    thresholds[str(th)]=[stable_cross(r['curve'],h,th) for r in rr]
   row={'dataset':d,'head':h,'E16':meta['datasets'][d]['E16'],'E50':meta['datasets'][d]['E50'],
    'validation_3000_mean':float(val.mean()),'validation_3000_std':float(val.std(ddof=1)),
    'final_test_mean':float(test.mean()),'final_test_std':float(test.std(ddof=1)),
    'last_validation_mean':float(np.mean([r['curve'][-1][h] for r in rr])),
    'end_steps':[r['step'] for r in rr],'plateau_seeds':[r['plateau'][h]['plateau'] for r in rr],
    'relative_last_window_change':[r['plateau'][h]['relative_change'] for r in rr],
    'steps_to_threshold':thresholds,'seed_test_mse':test.tolist(),'learned_encoder_image_energy':[r['encoder_retention'][h]['train_image_energy'] for r in rr]}
   row['T95_observed_improvement']=[]
   for r in rr:
    start=r['curve'][0][h];end=float(np.median([q[h] for q in r['curve'][-5:]]));target=end+.05*(start-end)
    row['T95_observed_improvement'].append(stable_cross(r['curve'],h,target) if end<start else None)
   row['validation_curve_average_3000']=float(np.mean([np.trapezoid([q[h] for q in r['curve'] if q['step']<=3000],[q['step'] for q in r['curve'] if q['step']<=3000])/3000 for r in rr]))
   summary.append(row)
 save_json(ROOT/'summary.json',summary)
 keys=['dataset','head','E16','E50','validation_3000_mean','validation_3000_std','final_test_mean','final_test_std','end_steps','plateau_seeds','T95_observed_improvement','validation_curve_average_3000','learned_encoder_image_energy']
 with (ROOT/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in summary])
 fig,axes=plt.subplots(5,2,figsize=(12,15),sharex=True,layout='constrained')
 steps=np.arange(0,3001,100)
 for i,d in enumerate(protocol['datasets']):
  rr=[r for r in records if r['dataset']==d]
  for h in ['x','eps']:
   vals=np.array([[next(p[h] for p in r['curve'] if p['step']==s) for s in steps] for r in rr]);mean=vals.mean(0);sd=vals.std(0,ddof=1)
   axes[i,1].plot(steps,mean,c=COLORS[h],label='Image target x' if h=='x' else 'Noise target epsilon')
   axes[i,1].fill_between(steps,mean-sd,mean+sd,color=COLORS[h],alpha=.15)
   trains=[]
   for r in rr:
    raw=list(csv.DictReader((ROOT/'runs'/f'{d}_s{r["seed"]}'/'training.csv').open()))
    values=np.array([float(row[h]) for row in raw[:3000]])
    assert len(values)==3000
    trains.append(values.reshape(-1,50).mean(1))
   trains=np.array(trains);xs=np.arange(25,3000,50)
   axes[i,0].plot(xs,trains.mean(0),c=COLORS[h],label='Image target x' if h=='x' else 'Noise target epsilon')
   axes[i,0].fill_between(xs,trains.mean(0)-trains.std(0,ddof=1),trains.mean(0)+trains.std(0,ddof=1),color=COLORS[h],alpha=.15)
  for j in range(2):
   axes[i,j].set_title(f'{LABELS[d]} | E16={meta["datasets"][d]["E16"]:.1%} | '+('Training (50-update means)' if j==0 else 'Validation (fixed noise)'))
   axes[i,j].set_ylabel('Target MSE');axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=8);axes[i,j].set_xlim(0,3000)
  lo=min(ax.get_ylim()[0] for ax in axes[i]);hi=max(ax.get_ylim()[1] for ax in axes[i])
  for ax in axes[i]:ax.set_ylim(max(0,lo),hi)
 axes[-1,0].set_xlabel('Optimizer updates');axes[-1,1].set_xlabel('Optimizer updates')
 fig.suptitle('Matched ViT: z = 0.5 x + 0.5 epsilon | bottleneck 16/192 | mean +/- SD, 3 seeds',fontsize=14)
 fig.savefig(ROOT/'training_validation_3000.png',dpi=180);fig.savefig(ROOT/'training_validation_3000.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
 for h in ['x','eps']:
  rows=[r for r in summary if r['head']==h and r['dataset']!='real_coco'];xs=[r['E16']*100 for r in rows]
  for ax,field in [(axes[0],'validation_3000'),(axes[1],'final_test')]:
   ax.errorbar(xs,[r[field+'_mean'] for r in rows],yerr=[r[field+'_std'] for r in rows],c=COLORS[h],marker='o',capsize=4,label='Image target x' if h=='x' else 'Noise target epsilon')
   if h=='x':
    for x,r in zip(xs,rows):ax.annotate('a='+r['dataset'][-1],(x,r[field+'_mean']),xytext=((5,-14) if x>90 else (4,9)),textcoords='offset points',fontsize=8,color=COLORS[h])
   r=next(r for r in summary if r['head']==h and r['dataset']=='real_coco')
   ax.errorbar([r['E16']*100],[r[field+'_mean']],yerr=[r[field+'_std']],color=COLORS[h],marker='*',markersize=12,capsize=4,linestyle='none',label='Real COCO ('+h+')')
 for ax in axes:
  ax.set_xlabel('Training PCA energy in top 16 directions (%)');ax.set_ylabel('Target MSE');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left');ax.set_xlim(0,101);ax.set_ylim(0,1.04)
 axes[0].set_title('Equal budget: validation after 3000 updates');axes[1].set_title('Final independent test: actual stopping steps recorded')
 fig.savefig(ROOT/'energy_vs_error.png',dpi=180);fig.savefig(ROOT/'energy_vs_error.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 xpos=np.arange(5);width=.34
 for j,h in enumerate(['x','eps']):
  rows=[next(r for r in summary if r['dataset']==d and r['head']==h) for d in protocol['datasets']]
  positions=xpos+(j-.5)*width
  axes[0].bar(positions,[r['final_test_mean'] for r in rows],width,yerr=[r['final_test_std'] for r in rows],capsize=3,color=COLORS[h],label=h)
  for pos,r in zip(positions,rows):
   values=[v for v in r['T95_observed_improvement'] if v is not None]
   if len(values)==3:axes[1].bar(pos,np.mean(values),width,yerr=np.std(values,ddof=1),capsize=3,color=COLORS[h],label=h if pos==positions[0] else None)
   else:axes[1].text(pos,50,'unreached',rotation=90,ha='center',fontsize=8)
 for ax in axes:
  ax.set_xticks(xpos,['a=0','a=1','a=2','a=3','COCO']);ax.grid(axis='y',alpha=.2);ax.legend()
 axes[0].set(ylabel='Independent test MSE',title='Final error (paired stopping steps)')
 axes[1].set(ylabel='Updates',title="T95 of each model's own observed improvement")
 fig.suptitle('T95 compares progress toward different error floors; read together with final error',fontsize=11)
 fig.savefig(ROOT/'speed_and_final_error.png',dpi=180);fig.savefig(ROOT/'speed_and_final_error.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
 for d in protocol['datasets']:
  energy=np.array(meta['datasets'][d]['train_cumulative_energy'])
  ax.plot(np.arange(1,193),energy*100,label=LABELS[d])
 ax.axvline(16,color='gray',lw=.8);ax.set(xlabel='Number of PCA directions',ylabel='Cumulative training energy (%)',xlim=(0,192),ylim=(0,101),title='Same total energy, different concentration');ax.legend();ax.grid(alpha=.2)
 fig.savefig(ROOT/'pca_energy.png',dpi=180);fig.savefig(ROOT/'pca_energy.pdf');plt.close(fig)
 fig,axes=plt.subplots(5,2,figsize=(11,13),layout='constrained')
 for i,d in enumerate(protocol['datasets']):
  rr=[r for r in records if r['dataset']==d]
  for j,h in enumerate(['x','eps']):
   for r in rr:
    steps=[p['step'] for p in r['curve']];val=[p[h] for p in r['curve']]
    axes[i,j].plot(steps,val,label=f'seed {r["seed"]}: end {r["step"]}',alpha=.8,lw=1)
   axes[i,j].set_title(LABELS[d]+' | '+h);axes[i,j].set_ylabel('Validation target MSE');axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=7)
 for ax in axes[-1]:ax.set_xlabel('Actual optimizer updates')
 fig.savefig(ROOT/'extended_validation_actual.png',dpi=160);fig.savefig(ROOT/'extended_validation_actual.pdf');plt.close(fig)
 # Display examples consistently; visualization clipping is not used in training.
 fig,axes=plt.subplots(5,6,figsize=(10,8),layout='constrained')
 for i,d in enumerate(protocol['datasets']):
  data=torch.load(ROOT/'data'/f'{d}.pt',weights_only=True)
  for j in range(6):
   im=data['validation'][j].permute(1,2,0).numpy();axes[i,j].imshow(np.clip(im/6+.5,0,1));axes[i,j].axis('off')
  axes[i,0].set_title(LABELS[d],fontsize=9,loc='left')
 fig.savefig(ROOT/'data_examples.png',dpi=160);plt.close(fig)
 lines=['固定0.5混合的ViT对照实验结果','',
  '输入：z=0.5x+0.5epsilon；图像按训练均值中心化并统一全局RMS=1，噪声为标准高斯。',
  '合成数据：相同高斯DCT系数，只改变频谱指数alpha=0,1,2,3；真实参照COCO。',
  '模型：32x32 RGB，patch8，192->16->96，2层3头ViT，247200参数；FP32，CPU。',
  '训练：每目标每数据3个种子；batch32，AdamW(lr2e-4, betas0.9/0.95, wd0)，梯度裁剪1。',
  '主要比较为3000次更新；此后成对延长至两个头均满足稳定标准或达到6000上限。',
  '稳定标准：最近两个连续500步窗口的验证MSE中位数相对变化绝对值<1%；只是操作性判据，不证明全局收敛。',
  '数据：训练4000，验证512（曲线及停止），独立测试488（终点评估）。同种子两个头严格匹配初始化、批次和噪声。',
  '', 'dataset | E16 | head | val@3000 mean +/- SD | final test mean +/- SD | end updates | stable seeds']
 for r in summary:lines.append(f'{r["dataset"]} | {r["E16"]:.2%} | {r["head"]} | {r["validation_3000_mean"]:.6f} +/- {r["validation_3000_std"]:.6f} | {r["final_test_mean"]:.6f} +/- {r["final_test_std"]:.6f} | {r["end_steps"]} | {r["plateau_seeds"]}')
 lines+=['','达到阈值的速度：需连续三个验证点均不高于阈值；单位更新步，None表示未达到，不填造数值。']
 for r in summary:lines.append(r['dataset']+' '+r['head']+' '+json.dumps(r['steps_to_threshold'])+'; T95='+str(r['T95_observed_improvement']))
 lines+=['T95为达到自身已观察误差下降量95%的步数，只作次要描述；两个目标误差下限不同，T95更小不等于最终效果更好。']
 lines+=['','读取结果的限制：最终终点步数可能不同，跨数据的主要比较应看固定3000步；种子标准差不是数据总体置信区间。',
  '真实COCO与高斯合成数据除能量分布外仍有其他统计差异；只有合成alpha扫描用于受控频谱比较。',
  '图中E16是训练数据PCA前16维能量，不等于训练ViT实际保留率；后者另存summary.json的learned_encoder_image_energy。',
  '能量集中与误差的关系仅限本ViT、m=16、t=0.5和所测数据；不等于流形证明或一般生成FID结论。']
 (ROOT/'RESULTS.txt').write_text('\n'.join(lines))
 save_json(ROOT/'VERIFIED.json',{'pairs':15,'models':30,'checkpoints_sha256_verified':True,'data_splits_disjoint':True,'curves_have_all_checks':True,'torch_version':torch.__version__})
 print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
