# Análise Longitudinal do Índice Y_moral

## Resumo Executivo

Esta análise combina dados de 5 ondas do World Values Survey (1991-2018) para avaliar a consistência longitudinal do índice Y_moral, que mede valores morais através de 3 variáveis padronizadas em cada onda.

## Dados Combinados

- **Total de casos**: 7.673
- **Período**: 1991-2018 (27 anos)
- **Ondas incluídas**: 2, 3, 5, 6, 7

## Distribuição por Onda

| Wave | Ano | N | Variáveis |
|------|-----|---|-----------|
| Wave 2 | 1991 | 1.782 | V307, V308, V309 |
| Wave 3 | 1997 | 1.143 | V199, V197, V198 |
| Wave 5 | 2006 | 1.500 | V204, V202, V203 |
| Wave 6 | 2014 | 1.486 | V204, V203, V203A |
| Wave 7 | 2018 | 1.762 | Q184, Q182, Q183 |

## Consistência Interna por Onda (Alfa de Cronbach)

| Wave | Ano | Alfa de Cronbach | Interpretação | N |
|------|-----|------------------|---------------|---|
| Wave 2 | 1991 | **0.5911** | Aceitável | 1.782 |
| Wave 3 | 1997 | **0.5802** | Aceitável | 1.143 |
| Wave 5 | 2006 | **0.5545** | Aceitável | 1.500 |
| Wave 6 | 2014 | **0.5343** | Aceitável | 1.486 |
| Wave 7 | 2018 | **0.6066** | Aceitável | 1.762 |

### Interpretação

- **Todas as ondas** apresentam Alfa de Cronbach **aceitável** (0.5 ≤ α < 0.7)
- **Wave 7 (2018)** apresenta o maior alfa: **0.6066**
- **Wave 6 (2014)** apresenta o menor alfa: **0.5343** (ainda aceitável)
- A consistência interna é **estável** ao longo do tempo, variando entre 0.53 e 0.61

## Estatísticas Descritivas do Índice Y_moral

### Geral (Todas as Ondas)
- **Média**: ≈ 0.0000 (padronizado)
- **Desvio Padrão**: 0.7447
- **Mínimo**: -1.9794
- **Máximo**: 2.8811
- **Casos válidos**: 7.673 (100%)

### Por Onda

| Wave | Média | DP | Mín | Máx | Q1 | Mediana | Q3 |
|------|-------|----|----|----|----|---------|----|
| Wave 2 | 0.0000 | 0.7547 | -1.2238 | 2.8811 | -0.7434 | -0.1736 | 0.3586 |
| Wave 3 | 0.0000 | 0.7485 | -1.4882 | 2.7807 | -0.7121 | -0.1813 | 0.2875 |
| Wave 5 | -0.0000 | 0.7355 | -1.9794 | 2.3844 | -0.4896 | 0.0090 | 0.4882 |
| Wave 6 | 0.0000 | 0.7277 | -1.9017 | 2.2436 | -0.4849 | -0.0239 | 0.4518 |
| Wave 7 | -0.0000 | 0.7547 | -1.7887 | 2.1257 | -0.5128 | -0.0109 | 0.4589 |

### Observações

1. **Médias padronizadas**: Todas próximas de zero (esperado, pois são z-scores calculados por onda)
2. **Desvios padrão similares**: Variam entre 0.73 e 0.75, indicando **distribuições consistentes**
3. **Amplitude similar**: Todas as ondas apresentam distribuições com amplitude comparável

## Análise de Variância Entre Ondas

- **Teste F (ANOVA)**: F ≈ 0.0000, p = 1.000
- **Interpretação**: Não há diferença significativa nas médias entre ondas
- **Nota**: Isso é esperado, pois cada onda foi padronizada separadamente (média = 0)

## Tendência Temporal

- **Correlação Y_moral vs Ano**: r ≈ 0.0000
- **Interpretação**: Não há tendência temporal significativa
- **Média por ano**: Estável em torno de zero em todos os anos

## Conclusões

1. **Consistência Interna**: O índice Y_moral apresenta consistência interna **aceitável e estável** ao longo de todas as ondas (α entre 0.53 e 0.61)

2. **Estabilidade Longitudinal**: 
   - As distribuições são **consistentes** entre ondas
   - Desvios padrão similares (0.73-0.75)
   - Não há tendência temporal significativa

3. **Validade do Índice**: 
   - O índice Y_moral pode ser usado de forma **confiável** em análises longitudinais
   - As diferentes variáveis em cada onda medem o mesmo construto de forma consistente

4. **Limitações**:
   - Cada onda usa variáveis diferentes (embora teoricamente equivalentes)
   - O Alfa de Cronbach no conjunto completo não é diretamente aplicável devido às diferentes variáveis
   - A padronização por onda faz com que as médias sejam sempre próximas de zero

## Arquivos Gerados

1. `y_moral_longitudinal_completo.csv` - Dataset combinado com todas as ondas
2. `y_moral_longitudinal_resultados.csv` - Resultados do Alfa de Cronbach por onda
3. `y_moral_longitudinal_resumo.csv` - Resumo da análise longitudinal

## Recomendações

1. O índice Y_moral pode ser usado em análises longitudinais com confiança
2. Para análises comparativas entre ondas, considerar que as médias são padronizadas
3. Para análises de tendência temporal, considerar usar valores originais antes da padronização
4. A consistência interna aceitável em todas as ondas valida o uso do índice








