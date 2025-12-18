# Análise WVS Wave 7 Brazil - Comparação por Religião

Este diretório contém scripts para analisar os respondentes da World Values Survey Wave 7 Brazil, separando-os por religião (Evangélicos, Católicos, Outros) e comparando as respostas às questões Q20 e Q21.

## Problema com o arquivo .sav

O arquivo `WVS_Wave_7_Brazil_Spss_v5.1.sav` parece ter um formato que não é totalmente compatível com as bibliotecas Python disponíveis (pyreadstat). 

## Soluções

### Opção 1: Converter para CSV no SPSS

1. Abra o arquivo `WVS_Wave_7_Brazil_Spss_v5.1.sav` no SPSS
2. Vá em `File > Save As`
3. Escolha o formato CSV
4. Salve como `WVS_Wave_7_Brazil_Spss_v5.1.csv` na mesma pasta
5. Execute o script: `python3 wvs_religion_analysis.py`

### Opção 2: Usar R para converter

Se você tiver R instalado:

```r
library(haven)
data <- read_sav("WVS_Wave_7_Brazil_Spss_v5.1.sav")
write.csv(data, "WVS_Wave_7_Brazil_Spss_v5.1.csv", row.names = FALSE)
```

### Opção 3: Usar SPSS Syntax

No SPSS, execute:

```spss
SAVE TRANSLATE OUTFILE='WVS_Wave_7_Brazil_Spss_v5.1.csv'
  /TYPE=CSV
  /MAP.
```

## Executando a análise

Após converter o arquivo para CSV, execute:

```bash
python3 wvs_religion_analysis.py
```

O script irá:
1. Carregar os dados
2. Identificar automaticamente as variáveis de religião, Q20 e Q21
3. Separar os respondentes em três grupos: Evangélicos, Católicos, Outros
4. Comparar as respostas Q20 e Q21 entre os grupos
5. Gerar tabelas de contingência e testes estatísticos
6. Criar visualizações
7. Salvar todos os resultados em arquivos CSV e PNG

## Variáveis esperadas

O script procura automaticamente por:
- **Religião**: Variáveis que contenham "Q289", "RELIG", ou "RELIGION"
- **Q20**: Variável exatamente chamada "Q20" ou similar
- **Q21**: Variável exatamente chamada "Q21" ou similar

Se as variáveis tiverem nomes diferentes, o script mostrará todas as variáveis disponíveis para você identificar manualmente.

## Resultados

O script gera os seguintes arquivos:

- `wvs_wave7_com_grupos_religiosos.csv`: Dados completos com a coluna `religiao_grupo` adicionada
- `q20_crosstab.csv`: Tabela de contingência Q20
- `q20_crosstab_percentual.csv`: Percentuais Q20 por grupo
- `q21_crosstab.csv`: Tabela de contingência Q21
- `q21_crosstab_percentual.csv`: Percentuais Q21 por grupo
- `comparacao_q20_q21_por_religiao.png`: Gráficos comparativos

## Dependências

```bash
pip install pandas numpy matplotlib seaborn scipy pyreadstat
```










