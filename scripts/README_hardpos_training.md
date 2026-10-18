# Treinar HardPos em outra máquina

Execute os notebooks a partir da raiz do repositorio. Os insumos versionados ficam em `runs/datasets/hardpos/` e `runs/datasets/sargolini/`. O treino HardPos usa o dataset e o mapeamento da primeira pasta. Instale as dependencias e execute todas as celulas de `navitran_vizz_reboot_gpt_squared_25_span.ipynb`, inclusive a celula final que registra a execucao ativa.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r reqs.txt
jupyter lab "navitran_vizz_reboot_gpt_squared_25_span.ipynb"
```

No Windows, ative `.venv\Scripts\Activate.ps1` em vez do comando `.` acima. O diretório de trabalho do kernel deve ser a raiz do repositório.

O notebook lê `runs/datasets/hardpos/ps_to_psid.pkl` e substitui a coluna `position_state_id` do Parquet pelos IDs desse arquivo antes de montar as sequências de treino. O pkl cobre todos os 947 PS observados no dataset e contém 110 estados adicionais, todos com IDs únicos. Para os PS observados, seus IDs coincidem com `ngroup()`; a coluna de IDs que já vem no Parquet usa outra convenção. Não use essa coluna para interpretar os pesos do treino. O `model_tag` inclui uma impressão digital do mapeamento para impedir que artefatos com convenções diferentes sejam misturados.

Todos os artefatos novos ficam em `artifacts/model_runs/<model_tag>/`:

| Pasta | Conteúdo |
| --- | --- |
| `csv/` | métricas, matrizes de similaridade, distribuições e embeddings |
| `full_fwd/` | snapshots de pré treino, épocas e pós treino |
| `checkpoints/` | pesos do modelo |
| `analysis/` | ativações e mapas espaciais |
| `generations/` | sequências geradas |

`run.json` registra a tag e o hash do mapeamento. A última célula registra a tag em `artifacts/model_runs/active.json`. O notebook `apd_vizz_gpt_squared_25 HardPos.ipynb` lê essa tag e o mapeamento salvo na execução. O arquivo `.gitignore` mantém os resultados de treino fora do Git; apenas código e insumos devem entrar no commit.

Para verificar o ambiente e o mapeamento sem rodar 1.000 épocas:

```bash
python -m unittest discover -s tests
python scripts/check_hardpos_inputs.py
```

A regeneracao das 138 imagens oficiais depende dos resultados do treino e das analises, alem dos insumos em `runs/`. Veja `scripts/README_official_images.md`.
