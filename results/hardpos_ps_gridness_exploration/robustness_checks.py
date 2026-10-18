from pathlib import Path
import json, pickle, sys, time, os
import numpy as np
import pandas as pd
from scipy import ndimage, sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(os.environ.get('HARDPOS_ANALYSIS_ROOT', Path(__file__).resolve().parent))
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(OUT))
import analyze_hardpos_spatial as base
N=100
REPEATS=100
SEED=20261010

def fast_score(field,ring):
    corr=base.autocorr2d(field)
    vals={}
    for angle in (30,60,90,120,150):
        turned=ndimage.rotate(corr,angle,reshape=False,order=1,mode="constant",cval=np.nan)
        good=ring & np.isfinite(corr) & np.isfinite(turned)
        if good.sum()<100:
            return np.nan
        x,y=corr[good],turned[good]
        if np.std(x)<1e-8 or np.std(y)<1e-8:
            return np.nan
        vals[angle]=np.corrcoef(x,y)[0,1]
    return min(vals[60],vals[120])-max(vals[30],vals[90],vals[150])

def main():
    t0=time.perf_counter()
    grid=np.load(OUT/"ps_grid_1001.npz")["ps_id"]
    table=pd.read_parquet(Path(os.environ.get('HARDPOS_ACTIVATIONS', ROOT/'runs/hardpos/analysis/HardPos_single_token_hidden_activations.parquet'))).sort_values("input_ps_id")
    names=[c for c in table if c.startswith("mlp_activation__d")]
    ff=table[names].to_numpy(np.float32)
    ps=grid.ravel()
    known=ps<947
    b=np.minimum(np.arange(1001)*N//1000,N-1)
    bins=(b[:,None]*N+b[None,:]).ravel()
    counts=sparse.coo_matrix((np.ones(known.sum(),np.float32),(bins[known],ps[known].astype(int))),shape=(N*N,947)).tocsr()
    counts.sum_duplicates()
    known_count=np.asarray(counts.sum(axis=1)).ravel()
    all_count=np.bincount(bins,minlength=N*N)
    coverage=known_count/all_count
    weights=sparse.diags(1/np.maximum(known_count,1))@counts
    valid=(coverage>=.95).reshape(N,N)
    yy,xx=np.indices((2*N-1,2*N-1))
    radius=np.hypot(xx-(N-1),yy-(N-1))
    ring=(radius>=10)&(radius<=40)
    def all_scores(matrix):
        maps=np.asarray(weights@matrix,dtype=np.float32).T.reshape(len(names),N,N)
        maps[:,~valid]=np.nan
        return np.array([fast_score(field,ring) for field in maps])
    observed=all_scores(ff)
    recorded=pd.read_csv(OUT/"exploratory_rotation_scores.csv")
    expected=recorded.set_index("feature").loc[names,"score_fixed_ring"].to_numpy()
    np.testing.assert_allclose(observed,expected,atol=1e-5)
    rng=np.random.default_rng(SEED)
    null_max=[]
    null_d11=[]
    for rep in range(REPEATS):
        shuffled=ff[rng.permutation(len(ff))]
        vals=all_scores(shuffled)
        null_max.append(float(np.nanmax(vals)))
        null_d11.append(float(vals[names.index("mlp_activation__d11")]))
        if (rep+1)%20==0:
            print("shuffle",rep+1,"elapsed",round(time.perf_counter()-t0,1),flush=True)
    null_max=np.asarray(null_max)
    null_d11=np.asarray(null_d11)
    pd.DataFrame({"repeat":np.arange(REPEATS),"max_ff_score":null_max,"d11_score":null_d11}).to_csv(OUT/"ps_shuffle_null_ff.csv",index=False)
    rows=[]
    maps=np.load(OUT/"all_dimension_maps_100.npz")
    all_names=maps["feature_names"].tolist()
    for name in ["mlp_activation__d11","key_projection__d08","mlp_activation__d16"]:
        ac=base.autocorr2d(maps["maps"][all_names.index(name)])
        for inner,outer in [(5,30),(10,30),(10,40),(15,40),(15,45),(20,40)]:
            base.R_IN,base.R_OUT=inner,outer
            _,profile=base.rotation_profile(ac)
            rows.append({"feature":name,"inner_radius":inner,"outer_radius":outer,"score":base.score(profile)})
    pd.DataFrame(rows).to_csv(OUT/"ring_sensitivity.csv",index=False)
    obsmax=float(np.nanmax(observed))
    pmax=float((1+np.sum(null_max>=obsmax))/(1+REPEATS))
    p11=float((1+np.sum(null_d11>=observed[names.index("mlp_activation__d11")]))/(1+REPEATS))
    result={"repeat_count":REPEATS,"seed":SEED,"observed_max_ff":obsmax,
            "observed_max_feature":names[int(np.nanargmax(observed))],
            "null_max_median":float(np.median(null_max)),"null_max_95pct":float(np.quantile(null_max,.95)),
            "empirical_p_max_ff":pmax,"empirical_p_d11_unadjusted":p11,
            "null_method":"Permute the 947 trained PS labels jointly across the 32 MLP activation dimensions; preserve PS region geometry and joint feature values.",
            "elapsed_seconds":time.perf_counter()-t0}
    (OUT/"robustness_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axs[0].hist(null_max,bins=18,color="#7296ba")
    axs[0].axvline(obsmax,color="#ad3030",lw=2,label=f"Observed {obsmax:.3f}")
    axs[0].set(xlabel="Maximum contrast among 32 MLP units",ylabel="Permutations")
    axs[0].legend()
    sens=pd.DataFrame(rows)
    for name,g in sens.groupby("feature"):
        axs[1].plot(range(len(g)),g.score,marker="o",label=name)
    axs[1].axhline(0,color="gray",lw=1)
    axs[1].set_xticks(range(6),[f"{a}-{b}" for a,b in [(5,30),(10,30),(10,40),(15,40),(15,45),(20,40)]],rotation=35)
    axs[1].set(xlabel="Inner-outer radius in 100 x 100 bins",ylabel="Rotation contrast")
    axs[1].legend(fontsize=8)
    fig.savefig(OUT/"robustness_checks.png",dpi=160)
    plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
