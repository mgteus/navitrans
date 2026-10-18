"""Atlas exaustivo em 25x25 para um treino HardPos nas épocas 0, 500 e 1000.

O notebook chama as funções deste módulo. Nada é executado no import.
"""
from __future__ import annotations
import hashlib
import json
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage, sparse

import analyze_notebook_grid25 as atlas25

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
TRAIN_NOTEBOOK = ROOT / 'navitran_vizz_reboot_gpt_squared_25_span.ipynb'
TAG = 'HardPos_embed16_context10_epochs1000_'
OUT = ROOT / 'results' / 'hardpos_atlas_evolucao' / 'embed16_context10_epochs1000'
CHECKPOINT_DIR = ROOT / 'artifacts/model_runs' / TAG / 'checkpoints'
CHECKPOINTS = {
    0: CHECKPOINT_DIR / f'gpt_model_{TAG}epoch_0000.pt',
    500: CHECKPOINT_DIR / f'gpt_model_{TAG}epoch_0500.pt',
    1000: CHECKPOINT_DIR / f'gpt_model_{TAG}.pt',
}
GRID = 25
SIGMA = .75
RING = (2.5, 10.0)
DIMENSIONS = 208
PS_COUNT = 947


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def check_inputs() -> dict[int, Path]:
    """Check all inputs before making any outputs; a partial training run fails here."""
    required = [TRAIN_NOTEBOOK, ROOT/'runs/datasets/hardpos/ps_to_psid.pkl', ROOT/'runs/datasets/hardpos/bert_trainset_squared_25.gzip',
                ROOT/'runs/hardpos/analysis/ps_grid_1001.npz', *CHECKPOINTS.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Aguarde o Run All do treino. Arquivos ausentes:\n'+'\n'.join(missing))
    return CHECKPOINTS.copy()


def load_model_class():
    """Use the exact architecture definitions from the training notebook."""
    nb = json.loads(TRAIN_NOTEBOOK.read_text(encoding='utf8'))
    namespace = {'torch': torch, 'nn': nn, 'math': math}
    for marker in ('class PositionalEncoding(nn.Module):', 'class TrajectoryBERT_HardPos(nn.Module):'):
        matches = [''.join(cell['source']) for cell in nb['cells']
                   if cell['cell_type']=='code' and ''.join(cell['source']).lstrip().startswith(marker)]
        if len(matches)!=1:
            raise ValueError(f'Definição da arquitetura não encontrada uma única vez: {marker}')
        exec(compile(matches[0],str(TRAIN_NOTEBOOK),'exec'),namespace)
    return namespace['TrajectoryBERT_HardPos']


def load_ps_ids() -> np.ndarray:
    with (ROOT/'runs/datasets/hardpos/ps_to_psid.pkl').open('rb') as stream:
        mapping = pickle.load(stream)
    train = pd.read_parquet(ROOT/'runs/datasets/hardpos/bert_trainset_squared_25.gzip',columns=['position_state'])
    ids = np.sort(train.position_state.map(mapping).dropna().unique().astype(np.int64))
    if not np.array_equal(ids,np.arange(PS_COUNT)):
        raise ValueError('Os PS vistos no treino não correspondem aos IDs 0..946.')
    return ids


def load_geometry():
    with np.load(ROOT/'runs/hardpos/analysis/ps_grid_1001.npz',allow_pickle=False) as z:
        ps_grid = z['ps_id']
    if ps_grid.shape!=(1001,1001):
        raise ValueError(f'Geometria inesperada: {ps_grid.shape}')
    ps = ps_grid.ravel()
    known = (ps>=0)&(ps<PS_COUNT)
    b = np.minimum(np.arange(1001)*GRID//1000,GRID-1)
    bins = (b[:,None]*GRID+b[None,:]).ravel()
    counts = sparse.coo_matrix((np.ones(int(known.sum()),np.float32),
                                (bins[known],ps[known].astype(int))),
                               shape=(GRID*GRID,PS_COUNT)).tocsr()
    known_count = np.asarray(counts.sum(axis=1)).ravel().reshape(GRID,GRID)
    all_count = np.bincount(bins,minlength=GRID*GRID).reshape(GRID,GRID)
    if (known_count<3).any():
        raise ValueError('Há células sem cobertura suficiente de PS conhecidos.')
    smoothed_count = ndimage.gaussian_filter(known_count.astype(float),SIGMA,mode='constant')
    return counts, known_count, all_count, smoothed_count


@torch.inference_mode()
def extract_activations(checkpoint: Path, model_class, ps_ids: np.ndarray,
                        batch_size: int=256) -> tuple[list[str],np.ndarray]:
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    expected = {
        'embedding.weight': (948,16),
        'position_embedding.pe': (10,16),
        'encoder.layers.0.linear1.weight': (32,16),
        'encoder.layers.0.linear2.weight': (16,32),
    }
    for key,shape in expected.items():
        if key not in state or tuple(state[key].shape)!=shape:
            raise ValueError(f'{checkpoint.name}: {key} não tem forma {shape}')
    model = model_class(948,16,1,1,32,dropout=.2,max_len=10,
                        pad_token_id=947,causal=True)
    model.load_state_dict(state,strict=True)
    model.eval()
    layer = model.encoder.layers[0]
    if len(model.encoder.layers)!=1 or layer.norm_first:
        raise ValueError('Arquitetura não coincide com a extração de 11 representações.')
    batches = {}
    for start in range(0,len(ps_ids),batch_size):
        x = torch.as_tensor(ps_ids[start:start+batch_size],dtype=torch.long).unsqueeze(1)
        token = model.embedding(x)
        h0 = model.position_embedding(token)
        query,key,value = F.linear(h0,layer.self_attn.in_proj_weight,
                                   layer.self_attn.in_proj_bias).chunk(3,dim=-1)
        attention,_ = layer.self_attn(h0,h0,h0,need_weights=False)
        after_attention = layer.norm1(h0+layer.dropout1(attention))
        mlp1 = layer.linear1(after_attention)
        activation = layer.activation(mlp1)
        mlp2 = layer.linear2(layer.dropout(activation))
        after_ffn = layer.norm2(after_attention+layer.dropout2(mlp2))
        if not torch.allclose(model.fc_out(after_ffn)[:,0,:],model(x)[:,0,:],
                              atol=1e-5,rtol=1e-4):
            raise AssertionError(f'{checkpoint.name}: decomposição diverge do forward.')
        current = {
            'token_embedding':token,'embedding_plus_position':h0,
            'query_projection':query,'key_projection':key,'value_projection':value,
            'attention_output':attention,'after_attention_block':after_attention,
            'mlp_linear1':mlp1,'mlp_activation':activation,'mlp_linear2':mlp2,
            'after_ffn_block':after_ffn,
        }
        for name,value in current.items():
            batches.setdefault(name,[]).append(value[:,0,:].cpu().numpy().astype(np.float32))
    names = []
    matrices = []
    for name,parts in batches.items():
        matrix = np.concatenate(parts,axis=0)
        for dim in range(matrix.shape[1]):
            names.append(f'{name}__d{dim:02d}')
        matrices.append(matrix)
    features = np.concatenate(matrices,axis=1)
    if features.shape!=(PS_COUNT,DIMENSIONS) or not np.isfinite(features).all():
        raise ValueError(f'Ativações inesperadas: {features.shape}')
    return names,features


def make_maps(features: np.ndarray, geometry):
    counts,known_count,all_count,smoothed_count = geometry
    sums = np.asarray(counts@features,dtype=np.float64).reshape(GRID,GRID,DIMENSIONS)
    raw,smoothed = atlas25.valid_maps(sums,known_count,smoothed_count)
    if not np.isfinite(raw).all() or not np.isfinite(smoothed).all():
        raise ValueError('Mapa com célula vazia: esperado 625/625 pintadas.')
    return raw,smoothed


def save_stage(epoch: int, ps_ids: np.ndarray, names: list[str],
               features: np.ndarray, raw: np.ndarray, smoothed: np.ndarray,
               geometry) -> pd.DataFrame:
    out = OUT/f'epoch_{epoch:04d}'
    out.mkdir(parents=True,exist_ok=True)
    table = pd.DataFrame(features,columns=names)
    table.insert(0,'input_ps_id',ps_ids)
    table.to_parquet(out/'activations_by_ps.parquet',index=False)
    _,known_count,all_count,_=geometry
    np.savez_compressed(out/'maps_25.npz',feature_names=np.asarray(names),
                        raw=raw,smoothed=smoothed,known_count=known_count,
                        all_count=all_count)
    scores = pd.DataFrame({'epoch':epoch,'feature':names,
                           'representation':[name.split('__d')[0] for name in names],
                           'G_raw':atlas25.score_all(raw),
                           'G_smoothed':atlas25.score_all(smoothed)})
    if not np.isfinite(scores[['G_raw','G_smoothed']].to_numpy()).all():
        raise ValueError('Gridness indefinido em pelo menos uma dimensão.')
    scores.to_csv(out/'gridness_208.csv',index=False)
    return scores


def color_ranges(maps: dict[int,tuple[np.ndarray,np.ndarray]]):
    shared=[]
    for j in range(DIMENSIONS):
        values=np.concatenate([field[j].ravel() for pair in maps.values() for field in pair])
        lo,hi=np.percentile(values,[2,98])
        shared.append((float(lo),float(hi if hi>lo else lo+1e-7)))
    return shared


def plot_atlases(names: list[str],maps: dict[int,tuple[np.ndarray,np.ndarray]],
                 scores: pd.DataFrame):
    ranges=color_ranges(maps)
    cmap=plt.get_cmap('viridis').copy()
    cmap.set_bad('#d8dddd')
    stage_names=list(dict.fromkeys(name.split('__d')[0] for name in names))
    for epoch,(raw,smoothed) in maps.items():
        out=OUT/f'epoch_{epoch:04d}'
        epoch_scores=scores[scores.epoch==epoch].set_index('feature')
        for stage in stage_names:
            indexes=[i for i,n in enumerate(names) if n.startswith(stage+'__d')]
            cols=4 if len(indexes)==16 else 8
            rows=math.ceil(len(indexes)/cols)
            for kind,field,column in [('raw',raw,'G_raw'),('smoothed',smoothed,'G_smoothed')]:
                fig,axes=plt.subplots(rows,cols,figsize=(2.55*cols,2.7*rows),
                                      constrained_layout=True)
                for ax,j in zip(np.ravel(axes),indexes):
                    ax.imshow(field[j],origin='lower',cmap=cmap,
                              vmin=ranges[j][0],vmax=ranges[j][1],interpolation='nearest')
                    ax.set_title(f'd{names[j].split("__d")[1]}  G={epoch_scores.loc[names[j],column]:+.2f}',fontsize=9)
                    ax.set(xticks=[],yticks=[])
                for ax in np.ravel(axes)[len(indexes):]:ax.axis('off')
                fig.suptitle(f'{stage} | época {epoch} | {kind} | 25×25 | 625/625 células',fontsize=12)
                fig.savefig(out/f'atlas_{stage}_{kind}.png',dpi=155)
                plt.close(fig)
    return ranges


def plot_comparisons(names,maps,scores,ranges):
    OUT.mkdir(parents=True,exist_ok=True)
    epochs=list(maps)
    final=scores[scores.epoch==max(epochs)].nlargest(5,'G_smoothed')
    score_index=scores.set_index(['epoch','feature'])
    cmap=plt.get_cmap('viridis')
    fig,axes=plt.subplots(5,3,figsize=(10.4,15.5),constrained_layout=True)
    for row,name in enumerate(final.feature):
        j=names.index(name)
        for col,epoch in enumerate(epochs):
            field=maps[epoch][1][j]
            ax=axes[row,col]
            ax.imshow(field,origin='lower',cmap=cmap,vmin=ranges[j][0],vmax=ranges[j][1])
            g=score_index.loc[(epoch,name),'G_smoothed']
            ax.set_title(f'{name}\népoca {epoch}: G={g:+.3f}',fontsize=9)
            ax.set(xticks=[],yticks=[])
    fig.savefig(OUT/'top5_mapas_tres_epocas.png',dpi=165)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.8,5.5),constrained_layout=True)
    for name in names:
        part=scores[scores.feature==name].sort_values('epoch')
        ax.plot(part.epoch,part.G_smoothed,color='#a6b9bc',alpha=.35,lw=.7)
    colors=plt.cm.tab10(np.linspace(0,.5,5))
    for color,name in zip(colors,final.feature):
        part=scores[scores.feature==name].sort_values('epoch')
        ax.plot(part.epoch,part.G_smoothed,marker='o',lw=2,label=name,color=color)
    ax.axhline(.5,color='#da8a25',ls='--',lw=.9)
    ax.set(xticks=epochs,xlabel='Época',ylabel='G suavizado (anel 2,5–10)',
           title='Gridness das mesmas 208 dimensões ao longo do treino')
    ax.legend(fontsize=8)
    fig.savefig(OUT/'gridness_evolucao.png',dpi=170)
    plt.close(fig)
    bins=np.arange(-1.4,1.401,.1)
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.4),sharex=True,sharey=True,constrained_layout=True)
    for ax,epoch in zip(axes,epochs):
        values=scores.loc[scores.epoch==epoch,'G_smoothed']
        ax.hist(values,bins=bins,color='#087e8b',edgecolor='white',linewidth=.5)
        ax.axvline(.5,color='#da8a25',ls='--')
        ax.set(title=f'Época {epoch}: máximo {values.max():+.3f}',xlabel='G suavizado',ylabel='Dimensões')
    fig.savefig(OUT/'histogramas_tres_epocas.png',dpi=170)
    plt.close(fig)
    by_layer=scores.groupby(['epoch','representation']).agg(
        max_G=('G_smoothed','max'),median_G=('G_smoothed','median')).reset_index()
    by_layer.to_csv(OUT/'gridness_por_camada.csv',index=False)
    pivot=by_layer.pivot(index='representation',columns='epoch',values='max_G')
    fig,ax=plt.subplots(figsize=(8.2,6.7),constrained_layout=True)
    image=ax.imshow(pivot.to_numpy(),cmap='coolwarm',vmin=-1,vmax=1,aspect='auto')
    ax.set(xticks=np.arange(len(pivot.columns)),xticklabels=pivot.columns,
           yticks=np.arange(len(pivot.index)),yticklabels=pivot.index,
           title='Maior G suavizado por representação e época',xlabel='Época')
    fig.colorbar(image,ax=ax,label='Maior G')
    fig.savefig(OUT/'gridness_por_camada.png',dpi=170)
    plt.close(fig)


def save_summary(scores,geometry,null_summary=None):
    _,known_count,all_count,_=geometry
    stage_rows=[]
    for epoch,part in scores.groupby('epoch',sort=True):
        highest=part.nlargest(5,'G_smoothed')
        stage_rows.append({'epoch':int(epoch),'max_G_smoothed':float(highest.G_smoothed.iloc[0]),
                           'max_feature':str(highest.feature.iloc[0]),
                           'positive_count':int((part.G_smoothed>0).sum()),
                           'above_0p5_count':int((part.G_smoothed>.5).sum()),
                           'top5':highest[['feature','G_raw','G_smoothed']].to_dict('records'),
                           'checkpoint':str(CHECKPOINTS[int(epoch)].name),
                           'checkpoint_sha256':sha256(CHECKPOINTS[int(epoch)])})
    summary={'model_tag':TAG,'training_notebook':TRAIN_NOTEBOOK.name,
             'training_notebook_sha256':sha256(TRAIN_NOTEBOOK),
             'input_protocol':'single PS token per forward, model.eval(), 947 trained PS',
             'model':{'embed_size':16,'hidden_dim':32,'context_len':10,'max_len':10,
                      'num_epochs':1000,'num_layers':1,'num_heads':1},
             'geometry':'all 1001x1001 coordinate positions, known PS only',
             'geometry_sha256':sha256(ROOT/'runs/hardpos/analysis/ps_grid_1001.npz'),
             'mapping_sha256':sha256(ROOT/'runs/datasets/hardpos/ps_to_psid.pkl'),
             'grid_size':GRID,'sigma_bins':SIGMA,'ring_radius_bins':list(RING),
             'dimensions':DIMENSIONS,'known_positions':int(known_count.sum()),
             'unknown_positions':int(all_count.sum()-known_count.sum()),
             'bins_with_values':int((known_count>0).sum()),
             'stages':stage_rows, 'null_control':null_summary}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return summary


def save_null_controls(features_by_epoch: dict[int,np.ndarray], scores: pd.DataFrame,
                       geometry, repeats: int=100, seed: int=20261013):
    """Joint PS permutations; use the same shuffles at each training stage."""
    counts,known_count,_,smoothed_count=geometry
    rng=np.random.default_rng(seed)
    permutations=[rng.permutation(PS_COUNT) for _ in range(repeats)]
    result={}
    for epoch,features in features_by_epoch.items():
        null_max=[]
        for perm in permutations:
            sums=np.asarray(counts@features[perm],dtype=np.float64).reshape(GRID,GRID,DIMENSIONS)
            _,field=atlas25.valid_maps(sums,known_count,smoothed_count)
            null_max.append(float(np.nanmax(atlas25.score_all(field))))
        null_max=np.asarray(null_max)
        if not np.isfinite(null_max).all():
            raise ValueError(f'Controle nulo indefinido na época {epoch}.')
        observed=float(scores.loc[scores.epoch==epoch,'G_smoothed'].max())
        exceed=int((null_max>=observed).sum())
        p=(1+exceed)/(repeats+1)
        stage_dir=OUT/f'epoch_{epoch:04d}'
        pd.DataFrame({'repeat':np.arange(repeats),'max_G_among_208':null_max}).to_csv(
            stage_dir/'null_max_208.csv',index=False)
        fig,ax=plt.subplots(figsize=(7.6,4.6),constrained_layout=True)
        ax.hist(null_max,bins=16,color='#9dbfc5',edgecolor='white')
        ax.axvline(observed,color='#bd3a37',lw=2,label=f'Máximo real {observed:+.3f}')
        ax.set(title=f'Época {epoch} | controle por {repeats} permutações conjuntas',
               xlabel='Maior G em 208 dimensões permutadas',ylabel='Repetições')
        ax.legend()
        fig.savefig(stage_dir/'null_max_208.png',dpi=170)
        plt.close(fig)
        result[str(epoch)]={'repeats':repeats,'seed':seed,'max_observed':observed,
                            'exceed_count':exceed,'empirical_p_max_208':p,
                            'null_median':float(np.median(null_max)),
                            'null_95pct':float(np.quantile(null_max,.95))}
    return result


def export_pdf():
    """Pack comparison charts and all 66 atlas plates into one review PDF."""
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    destination=OUT/'atlas_tres_epocas.pdf'
    lead=[OUT/'gridness_evolucao.png',OUT/'histogramas_tres_epocas.png',
          OUT/'gridness_por_camada.png',OUT/'top5_mapas_tres_epocas.png']
    plates=[]
    for epoch in CHECKPOINTS:
        plates.extend(sorted((OUT/f'epoch_{epoch:04d}').glob('atlas_*.png')))
        plates.append(OUT/f'epoch_{epoch:04d}'/'null_max_208.png')
    files=lead+plates
    missing=[str(path) for path in files if not path.is_file()]
    if missing or len(plates)!=69:
        raise FileNotFoundError(f'Figuras ausentes ({len(plates)} pranchas): {missing}')
    with PdfPages(destination) as pdf:
        for path in files:
            with Image.open(path) as im:
                pixels=np.asarray(im.convert('RGB'))
            height,width=pixels.shape[:2]
            size=(11.69,8.27) if width>height else (8.27,11.69)
            fig=plt.figure(figsize=size)
            ax=fig.add_axes([.03,.03,.94,.94])
            ax.imshow(pixels)
            ax.axis('off')
            pdf.savefig(fig,dpi=100)
            plt.close(fig)
    return destination
