from pathlib import Path
import hashlib, json, pickle, time, os
import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('HARDPOS_ANALYSIS_ROOT', Path(__file__).resolve().parent))
N = 100
MIN_COVERAGE = .95
R_IN, R_OUT = 10, 40

def autocorr2d(field):
    valid = np.isfinite(field).astype(float)
    x = np.nan_to_num(field, nan=0).astype(float)
    def cv(a,b):
        return fftconvolve(a, b[::-1,::-1], mode="full")
    count = cv(valid,valid)
    sx, sy = cv(x,valid), cv(valid,x)
    vx = cv(x*x,valid) - sx*sx/np.maximum(count,1)
    vy = cv(valid,x*x) - sy*sy/np.maximum(count,1)
    cov = cv(x,x) - sx*sy/np.maximum(count,1)
    with np.errstate(divide="ignore",invalid="ignore"):
        corr = cov/np.sqrt(np.maximum(vx,0)*np.maximum(vy,0))
    corr[(count<100)|(vx<1e-10)|(vy<1e-10)] = np.nan
    return np.clip(corr,-1,1).astype(np.float32)

def rotation_profile(corr):
    yy,xx = np.indices(corr.shape)
    c = corr.shape[0]//2
    r = np.hypot(xx-c,yy-c)
    ring = (r>=R_IN)&(r<=R_OUT)
    angles = np.arange(0,181,6)
    profile = np.full(len(angles),np.nan)
    for k,angle in enumerate(angles):
        rotated = ndimage.rotate(corr,angle,reshape=False,order=1,mode="constant",cval=np.nan)
        good = ring & np.isfinite(corr) & np.isfinite(rotated)
        if good.sum()>=100:
            a,b = corr[good],rotated[good]
            if np.std(a)>1e-8 and np.std(b)>1e-8:
                profile[k]=np.corrcoef(a,b)[0,1]
    return angles,profile

def score(profile):
    if not np.isfinite(profile[[5,10,15,20,25]]).all():
        return float("nan")
    return float(min(profile[10],profile[20])-max(profile[5],profile[15],profile[25]))

def synthetic_test():
    yy,xx=np.indices((N,N))
    field=np.zeros((N,N),float)
    for row in range(-2,8):
        for col in range(-2,8):
            cx=col*21+(row%2)*10.5
            cy=row*21*np.sqrt(3)/2
            field+=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*2.5**2))
    cases={"hexagonal":field,"noise":np.random.default_rng(42).normal(size=(N,N))}
    result={}
    fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for row,(label,arr) in enumerate(cases.items()):
        ac=autocorr2d(arr)
        angles,profile=rotation_profile(ac)
        result[label]=score(profile)
        axs[row,0].imshow(arr,origin="lower",cmap="viridis")
        axs[row,1].imshow(ac,origin="lower",cmap="coolwarm",vmin=-1,vmax=1)
        axs[row,2].plot(angles,profile,marker=".",ms=3)
        axs[row,2].set(xlim=(0,180),ylim=(-1,1),xlabel="Rotation (degrees)",ylabel="Pearson")
        axs[row,0].set_title(label)
        axs[row,1].set_title("2D autocorrelation")
        axs[row,2].set_title(f"Contrast = {result[label]:+.3f}")
        for ax in axs[row,:2]: ax.set(xticks=[],yticks=[])
    fig.savefig(OUT/"synthetic_validation.png",dpi=150)
    plt.close(fig)
    assert result["hexagonal"] > result["noise"]+.2
    return result

def load():
    with (ROOT/"runs/datasets/hardpos/ps_to_psid.pkl").open("rb") as f: mapping=pickle.load(f)
    table=pd.read_parquet(Path(os.environ.get('HARDPOS_ACTIVATIONS', ROOT/'runs/hardpos/analysis/HardPos_single_token_hidden_activations.parquet'))).sort_values('input_ps_id')
    names=[c for c in table if "__d" in c]
    assert table.input_ps_id.tolist()==list(range(947)) and len(names)==208
    features=table[names].to_numpy(np.float32)
    assert np.isfinite(features).all()
    apd=pd.read_parquet(ROOT/"runs/datasets/hardpos/apd_position_grid.parquet",columns=["X","Y","position_state"])
    ix=np.rint(apd.X.to_numpy()*1000).astype(int)
    iy=np.rint(apd.Y.to_numpy()*1000).astype(int)
    assert len(apd)==1001**2
    assert np.array_equal(ix,np.tile(np.arange(1001),1001))
    assert np.array_equal(iy,np.repeat(np.arange(1001),1001))
    ps=apd.position_state.map(mapping).to_numpy(np.int16)
    assert (ps>=0).all()
    return mapping,names,features,ps.reshape(1001,1001)

def aggregate(ps_grid,features):
    ps=ps_grid.ravel()
    known=ps<len(features)
    b=np.minimum(np.arange(1001)*N//1000,N-1)
    bins=(b[:,None]*N+b[None,:]).ravel()
    count=sparse.coo_matrix((np.ones(known.sum(),np.float32),
                            (bins[known],ps[known].astype(int))),
                            shape=(N*N,len(features))).tocsr()
    count.sum_duplicates()
    known_count=np.asarray(count.sum(axis=1)).ravel()
    total=np.bincount(bins,minlength=N*N)
    coverage=(known_count/total).reshape(N,N).astype(np.float32)
    maps=(np.asarray(count@features)/np.maximum(known_count[:,None],1)).T.reshape(features.shape[1],N,N)
    maps=maps.astype(np.float32)
    maps[:,coverage<MIN_COVERAGE]=np.nan
    return maps,coverage,np.bincount(ps,minlength=1057)

def trajectory_agreement(ps_grid,mapping):
    train=pd.read_parquet(ROOT/"runs/datasets/hardpos/bert_trainset_squared_25.gzip",columns=["pos","position_state"])
    xy=np.stack(train.pos.to_numpy()).astype(float)
    ix=np.rint(xy[:,0]*1000).astype(int)
    iy=np.rint(xy[:,1]*1000).astype(int)
    same=ps_grid[iy,ix]==train.position_state.map(mapping).to_numpy()
    return train,xy,same

def save_top_ps(mapping,names,features,area,train):
    inverse={v:k for k,v in mapping.items()}
    visits=np.bincount(train.position_state.map(mapping).to_numpy(),minlength=len(mapping))
    if os.environ.get('HARDPOS_ACTIVATIONS'):
        visits=pd.read_csv(OUT/'visits_by_ps.csv').sort_values('input_ps_id').n_visits.to_numpy()
    rows=[]
    for j,name in enumerate(names):
        rep,dim=name.split("__d")
        v=features[:,j]
        for rank,ps in enumerate(np.argsort(-v,kind="stable")[:5],1):
            rows.append(dict(feature=name,representation=rep,dimension=int(dim),
                             rank=rank,ps_id=int(ps),position_state=inverse[int(ps)],
                             activation=float(v[ps]),area_pixels=int(area[ps]),
                             trajectory_visits=int(visits[ps]),
                             constant_feature=bool(np.ptp(v)<1e-8)))
    pd.DataFrame(rows).to_csv(OUT/"top5_ps_by_dimension.csv",index=False)

def plot_coverage(ps_grid,coverage,xy,same):
    fig,axs=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    axs[0].imshow(ps_grid>=947,origin="lower",extent=(0,1,0,1),cmap="Reds",vmin=0,vmax=1)
    axs[0].set_title("PS without trained embedding (red)")
    axs[1].imshow(coverage,origin="lower",extent=(0,1,0,1),cmap="viridis",vmin=0,vmax=1)
    axs[1].set_title("Coverage in 100 x 100 bins")
    axs[2].scatter(xy[same,0],xy[same,1],s=.1,alpha=.15,color="steelblue")
    axs[2].scatter(xy[~same,0],xy[~same,1],s=1.5,alpha=.75,color="red")
    axs[2].set_title("Trajectory PS mismatch (red)")
    for ax in axs: ax.set(xlim=(0,1),ylim=(0,1),xlabel="x",ylabel="y",aspect="equal")
    fig.savefig(OUT/"coverage_and_ps_consistency.png",dpi=160)
    plt.close(fig)

def plot_atlases(maps,names,scores):
    cmap=plt.get_cmap("viridis").copy()
    cmap.set_bad("#d0d0d0")
    for rep in dict.fromkeys(n.split("__d")[0] for n in names):
        indexes=[i for i,n in enumerate(names) if n.startswith(rep+"__d")]
        cols=4 if len(indexes)==16 else 8
        rows=int(np.ceil(len(indexes)/cols))
        fig,axs=plt.subplots(rows,cols,figsize=(2.5*cols,2.8*rows),constrained_layout=True)
        for ax,j in zip(np.ravel(axs),indexes):
            arr=maps[j]
            lo,hi=np.nanpercentile(arr,[2,98])
            if hi<=lo: hi=lo+1e-7
            ax.imshow(arr,origin="lower",cmap=cmap,vmin=lo,vmax=hi,interpolation="nearest")
            ax.set_title(f"d{names[j].split('__d')[1]}  G={scores[j]:+.2f}",fontsize=9)
            ax.set(xticks=[],yticks=[])
        for ax in np.ravel(axs)[len(indexes):]: ax.axis("off")
        fig.suptitle(f"{rep}: independent color scale per map; gray = untrained PS")
        fig.savefig(OUT/f"atlas_{rep}.png",dpi=130)
        plt.close(fig)

def plot_examples(ps_grid,features,maps,names,corrs,profiles,scores):
    ff=[i for i,n in enumerate(names) if n.startswith("mlp_activation__d") and np.isfinite(scores[i])]
    other=[i for i,n in enumerate(names) if not n.startswith("mlp_activation__d") and np.isfinite(scores[i])]
    chosen=[max(ff,key=lambda i:scores[i]),max(other,key=lambda i:scores[i]),
            sorted(ff,key=lambda i:scores[i])[len(ff)//2]]
    lut=np.full(1057,np.nan,np.float32)
    cmap=plt.get_cmap("viridis").copy()
    cmap.set_bad("#d0d0d0")
    fig,axs=plt.subplots(3,3,figsize=(12,12),constrained_layout=True)
    for row,j in enumerate(chosen):
        lut[:947]=features[:,j]
        fine=lut[ps_grid]
        lo,hi=np.nanpercentile(fine,[2,98])
        if hi<=lo: hi=lo+1e-7
        axs[row,0].imshow(fine,origin="lower",cmap=cmap,vmin=lo,vmax=hi)
        axs[row,1].imshow(maps[j],origin="lower",cmap=cmap,vmin=lo,vmax=hi)
        axs[row,2].imshow(corrs[j],origin="lower",cmap="coolwarm",vmin=-1,vmax=1)
        for radius in (R_IN,R_OUT):
            axs[row,2].add_patch(plt.Circle((N-1,N-1),radius,fill=False,color="black",lw=1))
        for ax,title in zip(axs[row],["Exact 1001 x 1001 PS map","Area mean, 100 x 100",
                                      f"2D autocorrelation, G={scores[j]:+.3f}"]):
            ax.set_title(title)
            ax.set(xticks=[],yticks=[])
        axs[row,0].set_ylabel(names[j],fontsize=9)
    fig.savefig(OUT/"example_maps_and_autocorr.png",dpi=150)
    plt.close(fig)
    j=chosen[0]
    rotated=ndimage.rotate(corrs[j],60,reshape=False,order=1,mode="constant",cval=np.nan)
    angles,profile=profiles[j]
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,arr,title in zip(axs[:2],[corrs[j],rotated],["Original autocorrelation","Rotated by 60 degrees"]):
        ax.imshow(arr,origin="lower",cmap="coolwarm",vmin=-1,vmax=1)
        ax.set_title(title)
        ax.set(xticks=[],yticks=[])
    axs[2].plot(angles,profile,marker=".",ms=4)
    for angle in (30,60,90,120,150): axs[2].axvline(angle,color="gray",alpha=.3)
    axs[2].set(xlim=(0,180),ylim=(-1,1),xlabel="Rotation (degrees)",ylabel="Pearson")
    axs[2].set_title(f"{names[j]}: G={scores[j]:+.3f}")
    fig.savefig(OUT/"autocorr_rotation_example.png",dpi=150)
    plt.close(fig)
    return chosen

def main():
    start=time.perf_counter()
    mapping,names,features,ps_grid=load()
    maps,coverage,area=aggregate(ps_grid,features)
    train,xy,same=trajectory_agreement(ps_grid,mapping)
    save_top_ps(mapping,names,features,area,train)
    np.savez_compressed(OUT/"ps_grid_1001.npz",ps_id=ps_grid)
    np.savez_compressed(OUT/"all_dimension_maps_100.npz",
                        feature_names=np.asarray(names),maps=maps,coverage=coverage)
    synthetic=synthetic_test()
    corrs,profiles,scores,rows=[],[],[],[]
    for name,field in zip(names,maps):
        ac=autocorr2d(field)
        angles,profile=rotation_profile(ac)
        g=score(profile)
        corrs.append(ac)
        profiles.append((angles,profile))
        scores.append(g)
        rows.append(dict(feature=name,representation=name.split("__d")[0],
                         dimension=int(name.split("__d")[1]),score_fixed_ring=g,
                         corr_30=profile[5],corr_60=profile[10],corr_90=profile[15],
                         corr_120=profile[20],corr_150=profile[25],map_std=float(np.nanstd(field))))
    scores=np.asarray(scores)
    scored=pd.DataFrame(rows)
    scored.to_csv(OUT/"exploratory_rotation_scores.csv",index=False)
    if os.environ.get('HARDPOS_ACTIVATIONS'):
        baseline_path=Path(__file__).resolve().parent/'exploratory_rotation_scores.csv'
        baseline=pd.read_csv(baseline_path).set_index('feature')
        old_scores=baseline.loc[scored.feature,'score_fixed_ring'].to_numpy(float)
        comparison=scored[['feature','representation']].copy()
        comparison['G_single_token']=old_scores
        comparison['G_contextual_mean']=scored.score_fixed_ring.to_numpy(float)
        comparison['delta_G']=comparison.G_contextual_mean-comparison.G_single_token
        comparison.to_csv(OUT/'gridness_comparison_single_vs_context.csv',index=False)
    np.savez_compressed(OUT/"autocorrelograms_100.npz",
                        feature_names=np.asarray(names),maps=np.asarray(corrs))
    plot_coverage(ps_grid,coverage,xy,same)
    plot_atlases(maps,names,scores)
    chosen=plot_examples(ps_grid,features,maps,names,corrs,profiles,scores)
    checkpoint=ROOT/"runs/hardpos/checkpoints/gpt_model_HardPos_.pt"
    result=dict(checkpoint=checkpoint.name,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                embedding_dimensions=16,ff_hidden_dimensions=32,
                feature_count=len(names),trained_ps=len(features),
                all_environment_ps=len(mapping),untrained_ps=len(mapping)-len(features),
                grid_pixels=int(ps_grid.size),trained_grid_pixels=int((ps_grid<947).sum()),
                trained_grid_fraction=float((ps_grid<947).mean()),
                trajectory_samples=len(same),trajectory_ps_agreement=float(same.mean()),
                map_grid=N,min_bin_coverage=MIN_COVERAGE,
                valid_map_bin_fraction=float((coverage>=MIN_COVERAGE).mean()),
                ring_radius_bins=[R_IN,R_OUT],synthetic_scores=synthetic,
                example_features=[names[i] for i in chosen],
                score_valid_count=int(np.isfinite(scores).sum()),
                top_scores=scored.sort_values("score_fixed_ring",ascending=False).head(10)[["feature","score_fixed_ring"]].to_dict("records"),
                elapsed_seconds=time.perf_counter()-start)
    (OUT/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
