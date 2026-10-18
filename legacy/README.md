# Arquivo de codigo e notebooks

A raiz mantem os notebooks de uso corrente:

- Geracao dos datasets: `start_v0_bert_squared.ipynb` (HardPos) e `start_v0_bert_squared_sargolini.ipynb` (Sargolini).
- Treino: `navitran_vizz_reboot_gpt_squared_25_span.ipynb` e `navitran_vizz_reboot_gpt_squared_25_span_sargolini.ipynb`.
- Visualizacao oficial: `apd_vizz_gpt_squared_25 HardPos.ipynb` e `apd_vizz_gpt_squared_25 HardPos_Sargolini.ipynb`.
- Analises de ativacao HardPos: os notebooks `HardPos_*.ipynb` da raiz.

`legacy/notebooks/` guarda versoes anteriores, testes e variantes exploratorias. `legacy/python/` guarda scripts e implementacoes antigas que nao fazem parte do fluxo corrente. Esses arquivos foram movidos sem apagar seu conteudo.

O codigo compartilhado usado pelos notebooks correntes fica em `project_support/csv_runs.py`. Os scripts de execucao e renderizacao ficam em `scripts/`. Execute os notebooks a partir da raiz para manter os caminhos relativos de dados e saidas.
