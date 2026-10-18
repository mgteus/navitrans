"""Rebuild the HardPos-16 spatial atlas with notebook cell 63 settings.

Map source: 13,134 trajectory visits; 25x25 bins; min occupancy 3;
occupancy-normalized Gaussian with sigma 0.75 bins.
"""
from __future__ import annotations
import hashlib, json, pickle, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, ndimage
import analyze_hardpos_spatial as base
from robustness_checks import fast_score

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).resolve().parent
OUT=SOURCE/"notebook_grid25"
OUT.mkdir(exist_ok=True)
N=25
SIGMA=.75
MIN_OCC=3
REPEATS=100
SEED=20261013
R_IN,R_OUT=2.5,10.0
yy,xx=np.indices((2*N-1,2*N-1))
radius=np.hypot(xx-(N-1),yy-(N-1))
RING=(radius>=R_IN)&(radius<=R_OUT)
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
                     "axes.spines.right":False})

def score_all(maps):
    return np.array([fast_score(field,RING) for field in maps],float)

def valid_maps(sums,occupancy,smooth_occupancy):
    raw=np.divide(sums,occupancy[:,:,None],
                  out=np.full_like(sums,np.nan,dtype=float),
                  where=occupancy[:,:,None]>0)
    smooth_sum=ndimage.gaussian_filter(sums,sigma=(SIGMA,SIGMA,0),
                                       mode="constant")
    blurred=np.divide(smooth_sum,smooth_occupancy[:,:,None],
                      out=np.full_like(sums,np.nan,dtype=float),
                      where=smooth_occupancy[:,:,None]>0)
    blurred[occupancy<MIN_OCC,:]=np.nan
    return (raw.transpose(2,0,1).astype(np.float32),
            blurred.transpose(2,0,1).astype(np.float32))

def plot_atlases(names,raw,blurred,scores,subtitle="13.134 visitas"):
    cmap=plt.get_cmap("viridis").copy()
    cmap.set_bad("#d8dddd")
    paths=[]
    for stage in dict.fromkeys(n.split("__d")[0] for n in names):
        indexes=[j for j,n in enumerate(names) if n.startswith(stage+"__d")]
        cols=4 if len(indexes)==16 else 8
        rows=int(np.ceil(len(indexes)/cols))
        scales=[]
        for j in indexes:
            good=np.concatenate((raw[j][np.isfinite(raw[j])],
                                 blurred[j][np.isfinite(blurred[j])]))
            lo,hi=np.percentile(good,(2,98))
            if hi<=lo:hi=lo+1e-7
            scales.append((lo,hi))
        for condition,maps,score_col in (("raw",raw,"G_raw"),
                                         ("smoothed",blurred,"G_smoothed")):
            fig,axs=plt.subplots(rows,cols,figsize=(2.55*cols,2.7*rows),
                                 constrained_layout=True)
            for ax,j,(lo,hi) in zip(np.ravel(axs),indexes,scales):
                ax.imshow(maps[j],origin="lower",cmap=cmap,vmin=lo,vmax=hi,
                          interpolation="nearest")
                ax.set_title(f"d{names[j].split('__d')[1]}  G={scores.loc[j,score_col]:+.2f}",
                             fontsize=9)
                ax.set(xticks=[],yticks=[])
            for ax in np.ravel(axs)[len(indexes):]:ax.axis("off")
            fig.suptitle(f"{stage} | {condition} | grid 25x25 | "
                         f"{subtitle} | escala de cor pareada por dimensão",
                         fontsize=12)
            path=OUT/f"atlas_{stage}_{condition}.png"
            fig.savefig(path,dpi=155)
            plt.close(fig)
            paths.append(path.name)
    return paths

def plot_histograms(scores,top,control_scores,raw,blurred,names,
                    control_maps,null_max,p):
    bins=np.arange(-1.2,1.201,.1)
    fig,axs=plt.subplots(1,2,figsize=(13.6,4.8),constrained_layout=True,
                         sharex=True,sharey=True)
    for ax,col,title,color in [(axs[0],"G_raw","Discreto","#657785"),
                                (axs[1],"G_smoothed","Gauss sigma 0,75","#087e8b")]:
        vals=scores[col].to_numpy()
        ax.hist(vals,bins=bins,color=color,edgecolor="white",linewidth=.5)
        ax.axvline(0,color="#102239",lw=.8)
        ax.axvline(.5,color="#d48d24",ls="--")
        ax.set(title=f"{title}: 208 dimensões, {sum(vals>.5)} com G>0,5",
               xlabel="G (anel 2,5-10)",ylabel="Dimensões",xlim=(-1.2,1.2))
        ax.grid(axis="y",alpha=.17)
    fig.savefig(OUT/"hist_all_208.png",dpi=175)
    plt.close(fig)
    stages=list(dict.fromkeys(scores.representation))
    for no,group in enumerate((stages[:6],stages[6:]),1):
        fig,axs=plt.subplots(3,2,figsize=(13.7,8.2),constrained_layout=True)
        for ax,stage in zip(axs.flat,group):
            part=scores[scores.representation==stage]
            ax.hist(part.G_raw,bins=bins,histtype="step",lw=1.5,
                    color="#657785",label="Discreto")
            ax.hist(part.G_smoothed,bins=bins,color="#087e8b",alpha=.7,
                    label="Gauss 0,75")
            ax.axvline(0,color="#102239",lw=.6)
            ax.axvline(.5,color="#d48d24",ls="--",lw=.9)
            ax.set(title=f"{stage.replace('_',' ')} | n={len(part)}\n"
                         f"máx={part.G_smoothed.max():+.3f}",
                   xlim=(-1.2,1.2),xlabel="G",ylabel="Dimensões")
            ax.tick_params(labelsize=8)
            ax.title.set_fontsize(10)
            ax.grid(axis="y",alpha=.17)
        for ax in list(axs.flat)[len(group):]:ax.axis("off")
        h,l=axs.flat[0].get_legend_handles_labels()
        fig.legend(h,l,loc="upper center",ncol=2)
        fig.savefig(OUT/f"hist_stages_{no}.png",dpi=170)
        plt.close(fig)
    labels=top.feature.tolist()+list(control_scores)
    values=top.G_smoothed.tolist()+list(control_scores.values())
    fig,ax=plt.subplots(figsize=(12.6,5),constrained_layout=True)
    y=np.arange(len(labels))
    ax.barh(y,values,color=["#087e8b"]*5+["#657785","#da8a25"],height=.63)
    ax.set_yticks(y,labels)
    ax.invert_yaxis()
    ax.axvline(0,color="#102239",lw=.8)
    ax.axvline(.5,color="#a87520",ls="--")
    ax.set(xlim=(-.4,1.25),xlabel="G após sigma 0,75; anel 2,5-10")
    ax.grid(axis="x",alpha=.17)
    for k,v in enumerate(values):
        ax.text(v+(.02 if v>=0 else -.02),k,f"{v:+.3f}",
                va="center",ha="left" if v>=0 else "right")
    fig.savefig(OUT/"top5_vs_controls.png",dpi=175)
    plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14.2,7),constrained_layout=True)
    cmap=plt.get_cmap("viridis").copy();cmap.set_bad("#ddd")
    panel_maps=[blurred[names.index(top.feature.iloc[0])],
                control_maps["Ruído por PS"],control_maps["Grade hexagonal ideal"]]
    panel_names=[top.feature.iloc[0],*control_maps]
    panel_scores=[top.G_smoothed.iloc[0],*control_scores.values()]
    for k,(field,label,g) in enumerate(zip(panel_maps,panel_names,panel_scores)):
        lo,hi=np.nanpercentile(field,(2,98))
        axs[0,k].imshow(field,origin="lower",cmap=cmap,vmin=lo,vmax=hi)
        ac=base.autocorr2d(field)
        axs[1,k].imshow(ac,origin="lower",cmap="coolwarm",vmin=-1,vmax=1)
        for r in (R_IN,R_OUT):
            axs[1,k].add_patch(plt.Circle((N-1,N-1),r,fill=False,
                                          color="#222",lw=.8))
        axs[0,k].set_title(f"{label}\nG={g:+.3f}",fontsize=10)
        for ax in axs[:,k]:ax.set(xticks=[],yticks=[])
    axs[0,0].set_ylabel("Mapa suavizado")
    axs[1,0].set_ylabel("Autocorrelação 2D")
    fig.savefig(OUT/"top1_and_controls_maps.png",dpi=175)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,4.7),constrained_layout=True)
    ax.hist(null_max,bins=16,color="#9dbfc5",edgecolor="white")
    ax.axvline(scores.G_smoothed.max(),color="#bd3a37",lw=2,
               label=f"Máximo real {scores.G_smoothed.max():+.3f}")
    ax.set(xlabel="Maior G entre 208 dimensões permutadas",
           ylabel="Repetições",title="Nulo: 100 permutações conjuntas de PS")
    ax.legend()
    fig.savefig(OUT/"null_max_208.png",dpi=175)
    plt.close(fig)

def main():
    start=time.perf_counter()
    df=pd.read_parquet(ROOT/"runs/datasets/hardpos/bert_trainset_squared_25.gzip")
    with (ROOT/"runs/datasets/hardpos/ps_to_psid.pkl").open("rb") as f:mapping=pickle.load(f)
    pid=df.position_state.map(mapping).to_numpy(np.int32)
    assert len(df)==13134 and len(np.unique(pid))==947 and np.isfinite(pid).all()
    pos=np.stack(df.pos.to_numpy()).astype(float)[:,:2]
    assert np.isfinite(pos).all() and ((pos>=0)&(pos<=1)).all()
    ix=np.clip((pos[:,0]*N).astype(int),0,N-1)
    iy=np.clip((pos[:,1]*N).astype(int),0,N-1)
    bins=iy*N+ix
    occupancy=np.bincount(bins,minlength=N*N).reshape(N,N)
    smooth_occ=ndimage.gaussian_filter(occupancy.astype(float),SIGMA,mode="constant")
    table=pd.read_parquet(ROOT/"runs/hardpos/analysis/HardPos_single_token_hidden_activations.parquet").sort_values("input_ps_id")
    assert table.input_ps_id.tolist()==list(range(947))
    names=[c for c in table if "__d" in c]
    features=table[names].to_numpy(np.float32)
    assert len(names)==208
    counts=sparse.coo_matrix((np.ones(len(pid)),(bins,pid)),shape=(N*N,947)).tocsr()
    sums=np.asarray(counts@features).reshape(N,N,208)
    raw,blurred=valid_maps(sums,occupancy,smooth_occ)
    with np.load(ROOT/"runs/hardpos/analysis/HardPos_single_token_spatial_maps.npz",allow_pickle=False) as saved:
        assert np.array_equal(occupancy,saved["occupancy"]) and int(saved["min_occupancy"])==MIN_OCC
        for stage in saved["representation_names"].tolist():
            idx=[j for j,n in enumerate(names) if n.startswith(stage+"__d")]
            np.testing.assert_allclose(raw[idx],saved[f"{stage}_raw"],atol=1e-5,equal_nan=True)
            np.testing.assert_allclose(blurred[idx],saved[f"{stage}_smoothed"],atol=1e-5,equal_nan=True)
    score_raw,score_smooth=score_all(raw),score_all(blurred)
    scores=pd.DataFrame({"feature":names,"representation":[n.split("__d")[0] for n in names],
                         "G_raw":score_raw,"G_smoothed":score_smooth})
    assert np.isfinite(score_raw).all() and np.isfinite(score_smooth).all()
    scores.to_csv(OUT/"score_comparison_25.csv",index=False)
    np.savez_compressed(OUT/"maps_25.npz",feature_names=np.asarray(names),
                        raw=raw,smoothed=blurred,occupancy=occupancy)
    paths=plot_atlases(names,raw,blurred,scores)
    rng=np.random.default_rng(SEED)
    null_max=[]
    for rep in range(REPEATS):
        perm=rng.permutation(947)
        sum_perm=np.asarray(counts@features[perm]).reshape(N,N,208)
        _,map_perm=valid_maps(sum_perm,occupancy,smooth_occ)
        null_max.append(float(np.nanmax(score_all(map_perm))))
        if (rep+1)%10==0:
            print("permutation",rep+1,"elapsed",round(time.perf_counter()-start,1),flush=True)
    null_max=np.asarray(null_max)
    pd.DataFrame({"repeat":np.arange(REPEATS),"max_G_among_208":null_max}).to_csv(
        OUT/"ps_shuffle_null_max_208.csv",index=False)
    # One artificial PS unit: one fixed random scalar for each PS.
    random_ps=np.random.default_rng(SEED).normal(size=947)
    random_sums=np.bincount(bins,weights=random_ps[pid],minlength=N*N).reshape(N,N,1)
    _,noise_map=valid_maps(random_sums,occupancy,smooth_occ)
    # Ideal geometric grid is sampled at each real visit, then binned identically.
    hex_visit=np.zeros(len(pos),float)
    xp,yp=pos.T
    for row in range(-2,8):
        for col in range(-2,8):
            cx=col*.21+(row%2)*.105
            cy=row*.21*np.sqrt(3)/2
            hex_visit+=np.exp(-((xp-cx)**2+(yp-cy)**2)/(2*.025**2))
    hex_sums=np.bincount(bins,weights=hex_visit,minlength=N*N).reshape(N,N,1)
    _,hex_map=valid_maps(hex_sums,occupancy,smooth_occ)
    control_maps={"Ruído por PS":noise_map[0],"Grade hexagonal ideal":hex_map[0]}
    control_scores={name:float(fast_score(field,RING)) for name,field in control_maps.items()}
    top=scores.nlargest(5,"G_smoothed")
    observed=float(scores.G_smoothed.max())
    p=float((1+np.count_nonzero(null_max>=observed))/(REPEATS+1))
    plot_histograms(scores,top,control_scores,raw,blurred,names,control_maps,null_max,p)
    summary={"checkpoint":"runs/hardpos/checkpoints/gpt_model_HardPos_.pt",
             "checkpoint_sha256":hashlib.sha256((ROOT/"runs/hardpos/checkpoints/gpt_model_HardPos_.pt").read_bytes()).hexdigest(),
             "source_notebook":"navitran_vizz_reboot_gpt_squared_25_span.ipynb: cell 63",
             "source_map":"runs/hardpos/analysis/HardPos_single_token_spatial_maps.npz",
             "trajectory_file":"runs/datasets/hardpos/bert_trainset_squared_25.gzip",
             "visits":len(df),"trained_ps":947,"dimensions":208,
             "grid_size":N,"map_shape":[N,N],"occupied_bins":int((occupancy>0).sum()),
             "bins_with_min_occupancy":int((occupancy>=MIN_OCC).sum()),
             "min_occupancy":MIN_OCC,"gaussian_sigma_bins":SIGMA,
             "gaussian_sigma_environment":SIGMA/N,
             "ring_radius_bins":[R_IN,R_OUT],
             "ring_radius_environment":[R_IN/N,R_OUT/N],
             "top5":top[["feature","G_raw","G_smoothed"]].to_dict("records"),
             "raw_max_G":float(score_raw.max()),"smoothed_max_G":observed,
             "raw_over_0p5":int((score_raw>.5).sum()),
             "smoothed_over_0p5":int((score_smooth>.5).sum()),
             "smoothed_positive":int((score_smooth>0).sum()),
             "control_scores":control_scores,
             "control_methods":{"noise":"One iid N(0,1) scalar per trained PS; same visit raster and smoothing",
                                "hex":"Ideal geometric Gaussian lattice sampled at each visit; spacing 0.21 environment, peak sigma 0.025"},
             "null_repeats":REPEATS,"null_seed":SEED,
             "null_median":float(np.median(null_max)),
             "null_95pct":float(np.quantile(null_max,.95)),
             "null_at_least_observed":int(np.count_nonzero(null_max>=observed)),
             "empirical_p_max_208":p,
             "atlases":paths,
             "elapsed_seconds":time.perf_counter()-start}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
