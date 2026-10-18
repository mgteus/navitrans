# Regeneração das imagens oficiais

Execute na raiz do repositório, com o ambiente Python do projeto e os dados numéricos disponíveis:

```bash
.nav/bin/python scripts/render_official_images.py --output outputs/official-rerender
```

O comando executa código de plotagem dos notebooks, grava os PNGs somente na pasta indicada e confere os 138 caminhos e nomes de `imgs/oficiais`. Os PNGs arquivados servem para conferir os nomes, nunca como fonte dos pixels.

| Grupo | Quantidade | Código-fonte |
| --- | ---: | --- |
| HardPos fixed | 67 | `apd_vizz_gpt_squared_25 HardPos.ipynb`; dois gráficos auxiliares em `render_official_fixed.py` |
| Sargolini ambiente | 12 | `start_v0_bert_squared_sargolini.ipynb` |
| Sargolini GPT | 22 | Células de plotagem de `navitran_vizz_reboot_gpt_squared_25_span_sargolini.ipynb` |
| Sargolini HardPos e exportadas | 34 | `apd_vizz_gpt_squared_25 HardPos_Sargolini.ipynb` |
| Atlas FFW | 3 | `results/hardpos_ps_gridness_exploration/plot_ffw_atlases.py` |

As imagens GPT de Sargolini usam métricas, matrizes de similaridade e estados `full_fwd` já calculados; o script não repete o treino de 2.000 épocas. Os três atlas FFW, embora arquivados em `sargolini/`, são calculados a partir de ativações HardPos.

## Entradas necessárias

A execução depende dos arquivos numéricos lidos pelos notebooks: datasets Parquet `runs/datasets/hardpos/bert_trainset_squared_25.gzip` e `runs/datasets/sargolini/sargolini_trainset_squared_25.gzip`, `runs/datasets/hardpos/ps_to_psid.pkl`, matrizes de similaridade, distribuições de probabilidade e logits, estados `full_fwd_*` associados a `model_tag`, métricas de treino, ativações HardPos e `runs/hardpos/analysis/ps_grid_1001.npz`. Muitos desses arquivos estão fora do Git ou cobertos por `.gitignore`. Assim, o comando foi validado nesta área de trabalho, mas ainda não é reproduzível a partir de um clone limpo. É necessário versionar ou publicar esses insumos numéricos com identificação de versão para cumprir esse requisito.

## Diferenças observadas na revisão

Os nomes e tipos de gráfico são regenerados, mas alguns valores e arranjos visuais diferem do arquivo oficial:

- Os 12 mapas de similaridade por PS de `fixed/` podem mudar bastante entre execuções porque usam os embeddings da tag ativa. Os CSVs `HardPos_` arquivados não correspondem aos valores preservados nas saídas antigas do notebook; a origem numérica exata dos mapas oficiais ainda não foi localizada.
- Os dois PNGs oficiais `celula_151_saida_01.png` e `celula_152_saida_01.png` são idênticos entre si no arquivo e nas saídas salvas do notebook. Na execução atual, as células 151 e 152 usam respectivamente pontos de mudança e todos os pontos, produzindo mapas distintos. A trajetória que gerou o PNG oficial não está identificada.
- Outros gráficos `fixed/` também variam com os resultados numéricos locais. Diferenças de resolução e bytes de PNG são esperadas na nova renderização.

Não substitua esses resultados pela cópia dos PNGs arquivados: isso esconderia as diferenças de origem dos dados.

### Diagnóstico dos mapas de similaridade HardPos

O fluxo esperado é: o treinamento calcula os vetores de embedding dos PS e grava a matriz de cossenos em CSV; o notebook de visualização combina cada linha dessa matriz com as posições dos PS e suaviza o mapa. Os IDs e a geometria não precisam mudar para que as cores mudem: as cores dependem dos pesos de embedding.

Há evidência numérica de que o CSV atual não é o mesmo usado para salvar as saídas do notebook de visualização. Ambos têm forma 948×948, mas a similaridade pré-treino PS 0–PS 1 aparece como -0,096624 na saída salva e +0,044635 no CSV `cosine_similarity_matrix_gpt_pre_train_HardPos_.csv`. Na época 199, os valores correspondentes são -0,013989 e +0,111174. O notebook de visualização procura o prefixo `HardPos_`, enquanto o notebook de treinamento `navitran_vizz_reboot_gpt_squared_25_span.ipynb` usa atualmente o prefixo `HardPos_embed16_context10_epochs1000_`. Nenhum dos 317 CSVs de similaridade encontrados na área de trabalho contém essas primeiras linhas originais. A causa exata da substituição ou perda dos CSVs originais ainda não foi comprovada.

## Organização dos CSVs

Os CSVs historicos permanecem apenas no arquivo local `legacy/csv/` e nao entram no Git. Os novos CSVs sao gravados em `artifacts/model_runs/<model_tag>/csv/`.

Os notebooks de treino escrevem novos CSVs em `artifacts/model_runs/<model_tag>/csv/` e novos snapshots `full_fwd` em `artifacts/model_runs/<model_tag>/full_fwd/`. Cada execução tem `run.json` com a tag. Ao terminar, os treinos HardPos e Sargolini registram a execução em `artifacts/model_runs/active.json`. Os dois notebooks de visualização leem essa tag e `load_csv_path` rejeita nomes de CSV com outra tag. A verificação é de identidade da execução, sem exigir números iguais aos de uma rodada anterior.

Quando o arquivo local `legacy/` estiver disponivel, `scripts/activate_legacy_run.py` pode copiar resultados historicos para `artifacts/model_runs/`. Os notebooks de treino, geracao de datasets e visualizacao nao dependem desse arquivo:

```bash
.nav/bin/python scripts/activate_legacy_run.py --group fixed --model-tag HardPos_embed16_context10_epochs1000_
.nav/bin/python scripts/activate_legacy_run.py --group sargolini --model-tag HardPos_Sargolini
```

Neste workspace, a tag HardPos ativa é `HardPos_embed16_context10_epochs1000_` e a Sargolini é `HardPos_Sargolini`. Os CSVs antigos continuam preservados em `legacy/`. Os demais notebooks com gravações CSV usam `artifacts/notebooks/<notebook>/csv/` ou suas pastas de análise já existentes.

## Organização dos snapshots `full_fwd`

Os 200 arquivos `full_fwd*` que estavam na raiz foram preservados, com os mesmos nomes e bytes, em `legacy/full_fwd/<model_tag>/`. O inventário está em `legacy/full_fwd/MANIFEST.tsv`. Os arquivos binários legados continuam locais e ignorados pelo Git. A ativação das tags HardPos e Sargolini copiadas para `artifacts/model_runs/` mantém a visualização e a regeneração das imagens funcionais sem retornar arquivos à raiz. Novas execuções gravam os snapshots diretamente na pasta da sua tag; a leitura recusa nomes pertencentes a outra tag.

Os insumos versionados de HardPos e Sargolini estao em `runs/datasets/`, incluindo os dois datasets de treino, o mapeamento HardPos, a base espacial Sargolini e os arquivos auxiliares. Em `runs/hardpos/analysis/`, apenas a tabela de ativacoes e a grade de PS usadas pelos tres atlas FFW entram no Git. Os checkpoints historicos `.pt` e os demais intermediarios ficam locais e ignorados; analises que usam esses pesos exigem treino ou transferencia para outra maquina. Os demais arquivos antigos permanecem apenas no arquivo local `legacy/`.

Os demais Parquet e NPZ de analise, os checkpoints e os binarios legados ignorados continuam locais. Um clone limpo ainda precisa executar o treino e gerar os CSVs e snapshots associados ao model_tag para reproduzir todos os graficos oficiais.

## Identidade PS→ID do HardPos

O treino novo lê `runs/datasets/hardpos/ps_to_psid.pkl` da raiz e salva uma cópia com hash na pasta de sua execução. O `model_tag` inclui a impressão digital desse mapeamento. As visualizações HardPos usam o mapeamento da tag ativa, não recriam IDs com `ngroup()`. A execução legada `HardPos_embed16_context10_epochs1000_` usa o mapeamento compacto de 947 PS que corresponde às suas matrizes de 948 colunas. O pkl da raiz contém os mesmos IDs para esses 947 PS e mais 110 estados não observados no dataset. A coluna `position_state_id` gravada no Parquet segue outra convenção e é substituída no treino. Para treinar em outro computador, siga `scripts/README_hardpos_training.md`.
