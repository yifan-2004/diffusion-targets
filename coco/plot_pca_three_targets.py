"""Offline x / epsilon / v target diagnostics, using the existing COCO PCA basis.

No model training. All curves use the same image PCA directions, never a
separately optimized basis for each target. v = x - epsilon for z_t = tx+(1-t)e.
"""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'prepared'
OUT = ROOT / 'coco_pca_three_targets'

def patches(x):
    return x.unfold(2,8,8).unfold(3,8,8).permute(0,2,3,1,4,5).reshape(-1,192)

def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    with np.load(ROOT/'coco_data/coco_val2017_preprocessed_32.npz',allow_pickle=False) as c:
        images=torch.from_numpy(c['images'].astype(np.float32)/127.5-1).permute(0,3,1,2).contiguous()
        lookup={n:i for i,n in enumerate(c['names'].tolist())}
    split=json.loads((BASE/'split.json').read_text())
    ds=torch.load(BASE/'data_summary.pt',weights_only=True)
    state=torch.load(BASE/'pca_state.pt',weights_only=True)
    U=state['eigenvectors'].double();mu=state['mean'].double()
    train=(images[[lookup[n] for n in split['train']]]-ds['train_mean'])/ds['scale']
    val=(images[[lookup[n] for n in split['heldout']]]-ds['train_mean'])/ds['scale']
    del images
    assert len(train)==4000 and len(val)==1000
    projected={name:patches(x).double()@U-mu@U for name,x in [('train',train),('heldout',val)]}
    image_energy={name:q.square().mean(0) for name,q in projected.items()}
    assert torch.allclose(image_energy['train'],state['eigenvalues'].double(),atol=1e-7)
    assert torch.allclose(image_energy['heldout'],torch.diag(U.T@state['heldout_second_moment']@U),atol=1e-7)
    seeds=[36290,36291,36292]
    energy={name:{'x':[], 'eps':[], 'v':[]} for name in projected}
    for name,q in projected.items():
        for seed in seeds:
            # White noise in pixel coordinates; fixed independent realizations.
            e=torch.randn((len(q),192),generator=torch.Generator().manual_seed(seed+(100000 if name=='heldout' else 0)),dtype=torch.float32).double()@U
            energy[name]['x'].append(image_energy[name].numpy())
            energy[name]['eps'].append(e.square().mean(0).numpy())
            energy[name]['v'].append((q-e).square().mean(0).numpy())
    result={'protocol':{'dataset':'COCO val2017','train_images':4000,'heldout_images':1000,
        'resolution':32,'patch_size':8,'patch_dimension':192,'normalization':'saved training pixel mean and RMS',
        'basis':'existing training-image PCA eigenvectors; shared across x, epsilon and v',
        'velocity_definition':'v = x - epsilon', 'noise':'independent N(0,I), pixel-space float32',
        'noise_seeds':seeds,'model_training':False,
        'centering':'x and v subtract saved training patch mean; epsilon has zero population mean',
        'uncertainty':'three noise realizations, not three newly trained model seeds'},'data':{}}
    for name in energy:
        result['data'][name]={}
        for target,rows in energy[name].items():
            rows=np.stack(rows);cum=np.cumsum(rows,axis=1)/rows.sum(axis=1,keepdims=True)
            result['data'][name][target]={'direction_energy_mean':rows.mean(0).tolist(),
                'direction_energy_std':rows.std(0,ddof=1).tolist(),
                'cumulative_mean':cum.mean(0).tolist(),'cumulative_std':cum.std(0,ddof=1).tolist(),
                'top12_fraction':float(cum[:,11].mean()),
                'dimensions_for_90_percent':[int(np.searchsorted(c,.9)+1) for c in cum]}
    result['population_reference']={'noise_direction_energy':[1.]*192,
        'v_direction_energy_train':(state['eigenvalues']+1).tolist(),
        'identity':'Cov(x-epsilon) = Cov(x) + I, for independent unit white noise'}
    (OUT/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    font=Path('/System/Library/AssetsV2/com_apple_MobileAsset_Font8/53fe5be564086fefc7523ccd0a31200acf92e0e5.asset/AssetData/STHEITI.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'pdf.fonttype':42})
    colors={'x':'#2364AA','eps':'#D1495B','v':'#23836E'}
    labels={'x':'图像 x','eps':'白噪声 ε','v':'速度 v = x − ε'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for target in ['x','eps','v']:
        left=result['data']['train'][target];right=result['data']['heldout'][target]
        axes[0].semilogy(np.arange(1,193),left['direction_energy_mean'],color=colors[target],label=labels[target],lw=2)
        cm=np.r_[0,right['cumulative_mean']]*100;cs=np.r_[0,right['cumulative_std']]*100
        axes[1].plot(np.arange(193),cm,color=colors[target],label=labels[target],lw=2)
        axes[1].fill_between(np.arange(193),cm-cs,cm+cs,color=colors[target],alpha=.15)
    axes[0].set(xlabel='训练图像 PCA 主成分序号',ylabel='各方向能量（对数坐标）',title='目标在同一 PCA 基中的能量谱')
    axes[1].set(xlabel='保留的图像主成分数 m',ylabel='累计保留能量（%）',title='留出集：逐项保留相同方向',ylim=(0,103))
    axes[1].axvline(12,color='#999999',ls=':',lw=1)
    notes='12 维保留能量：\n'+'\n'.join(f"{labels[t]}：{result['data']['heldout'][t]['top12_fraction']*100:.1f}%" for t in ['x','v','eps'])
    axes[1].text(.52,.20,notes,transform=axes[1].transAxes,fontsize=11,bbox={'facecolor':'white','edgecolor':'#dddddd','alpha':.95,'boxstyle':'round,pad=.5'})
    for ax in axes:
        ax.grid(alpha=.18);ax.legend(loc='lower left' if ax is axes[0] else 'upper left',fontsize=10)
    fig.savefig(OUT/'pca_x_eps_v.png',dpi=240)
    fig.savefig(OUT/'pca_x_eps_v.pdf')
    plt.close(fig)
    (OUT/'README.txt').write_text('三目标 PCA 诊断：与既有实验相同的 COCO 划分、归一化和图像 PCA 基。\n'
        '左：训练集各 PCA 方向的目标能量。右：留出集按相同方向保留的累计能量。\n'
        'v=x−ε；ε 为独立单位高斯噪声。三个噪声种子取均值，右图阴影为一个样本标准差。\n'
        '这是目标能量诊断，不是新增 v 模型训练，也不是学习到的编码器子空间诊断。\n'
        '速度谱使用共同图像 PCA 基，不是重新对 v 做排序；在总体上 Cov(v)=Cov(x)+I。\n')
    print(json.dumps({t:result['data']['heldout'][t] | {'direction_energy_mean':None,'direction_energy_std':None,'cumulative_mean':None,'cumulative_std':None} for t in ['x','eps','v']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
