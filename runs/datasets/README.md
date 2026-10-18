# Insumos dos notebooks

Execute os notebooks a partir da raiz do repositorio.

- `hardpos/`: dataset de treino HardPos, grade APD compacta e mapeamento `ps_to_psid.pkl`.
- `sargolini/`: dataset de treino Sargolini e base espacial completa para os mapas.
- `shared/`: configuracoes de objetos e pistas usadas pelos dois geradores.

Os notebooks `start_v0_bert_squared*.ipynb` gravam os datasets nestas pastas. Os notebooks `navitran_vizz_reboot_gpt_squared_25_span*.ipynb` leem daqui. Os arquivos selecionados estao no Git; resultados de treino e experimentos em outras pastas de `runs/` ficam locais.

Os dois checkpoints historicos em `runs/hardpos/checkpoints/` permanecem apenas locais: arquivos `.pt` nao entram no Git. As analises que dependem deles exigem treino ou transferencia dos pesos para outra maquina. Para os tres atlas FFW oficiais, o Git inclui somente a tabela de ativacoes e a grade de PS em `runs/hardpos/analysis/`. Outros resultados intermediarios de analise ficam locais. Novos checkpoints do treino sao gravados em `artifacts/model_runs/<model_tag>/checkpoints/`.
