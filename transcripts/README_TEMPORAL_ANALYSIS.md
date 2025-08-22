# Análise Temporal de Debates Presidenciais (2002-2022)

## 📋 Visão Geral

Este projeto realiza uma análise temporal completa dos temas discutidos em debates presidenciais brasileiros ao longo de 20 anos (2002-2022), identificando tendências, mudanças e padrões sociopolíticos na evolução do discurso político.

## 🎯 Objetivos

- **Análise Temporal**: Identificar como os temas ganharam ou perderam relevância ao longo do tempo
- **Detecção de Tendências**: Encontrar padrões de crescimento e declínio temático
- **Contextualização Histórica**: Relacionar mudanças temáticas com eventos históricos
- **Insights Políticos**: Fornecer recomendações baseadas em dados para candidatos, pesquisadores e cidadãos

## 📊 Dados Analisados

### Período: 2002-2022
- **Total de debates**: 14
- **Anos com debates**: 4 (2002, 2014, 2018, 2022)
- **Média de debates por ano**: 3.5

### Arquivos de Debate Analisados:
- **2002**: 7 debates (Bandeirantes - 1º turno completo + partes)
- **2014**: 1 debate (Globo - 2º turno)
- **2018**: 2 debates (Globo 1º turno + RedeTV 1º turno)
- **2022**: 4 debates (Bandeirantes - 1º e 2º turnos)

## 🔍 Metodologia

### 1. Extração de Temas
Utilizamos um dicionário temático com 9 categorias principais (baseado no themes.py original):
- **Educação**: ensino, escola, universidade, professor, aluno, etc.
- **Economia**: inflação, emprego, salário, PIB, crescimento, etc.
- **Armas**: violência, crime, polícia, segurança, etc.
- **Religião**: igreja, Deus, evangélicos, culto, etc.
- **Moralidade/Sexualidade**: família, valores, LGBT, aborto, etc.
- **Democracia**: eleições, voto, partidos, governo, etc.
- **Corrupção**: Lava Jato, mensalão, propina, etc.
- **Fake News**: desinformação, mentira, propaganda, etc.
- **Saúde**: SUS, hospital, vacina, pandemia, etc.

### 2. Processamento de Texto
- Normalização (remoção de acentos, conversão para minúsculas)
- Contagem de ocorrências por tema
- Cálculo de frequência por 10.000 palavras
- Agrupamento por ano e cálculo de médias

### 3. Análise Temporal
- Correlação com o tempo para identificar tendências
- Cálculo de variação percentual entre períodos
- Identificação de temas emergentes e em declínio

## 📈 Principais Descobertas

### 🚀 Temas Emergentes (Crescimento Significativo)
1. **Corrupção**: +2.659% (correlação: 0.811)
2. **Saúde**: +657% (correlação: 0.780)
3. **Fake News**: +602% (correlação: 0.733)

### 📉 Temas em Declínio (Queda Significativa)
1. **Economia**: -67.0% (correlação: -0.896)

### ⚖️ Temas Estáveis
- **Democracia**: -39.9% (correlação: -0.386)
- **Educação**: -17.9% (correlação: -0.077)
- **Religião**: +20.6% (correlação: 0.236)
- **Moralidade/Sexualidade**: +10.9% (correlação: 0.341)

## 🎯 Padrões Sociopolíticos Identificados

### 1. **Polarização e Valores**
- Crescimento consistente de temas relacionados a valores e identidade
- Religião, moralidade e fake news ganharam relevância
- Indica crescente polarização política

### 2. **Mudança no Foco Econômico**
- Temas econômicos tradicionais perderam relevância
- Mudança de foco para questões mais domésticas e identitárias

### 3. **Saúde e Bem-estar**
- Saúde emergiu como tema central (+657%)
- Provavelmente influenciado pela pandemia de COVID-19

### 4. **Corrupção como Tema Central**
- Corrupção se tornou um dos temas mais relevantes (+2.659%)
- Reflete escândalos como Lava Jato e crescente preocupação com transparência

## 📅 Contexto Histórico por Ano

### 2002
- **Contexto**: Eleição que levou Lula ao poder
- **Foco**: Economia e democracia
- **Temas principais**: Democracia (227.4), Economia (199.6), Educação (34.7)

### 2014
- **Contexto**: Reeleição de Dilma, escândalos de corrupção emergindo
- **Foco**: Democracia e corrupção
- **Temas principais**: Democracia (336.8), Economia (82.2), Educação (62.8)

### 2018
- **Contexto**: Eleição de Bolsonaro, polarização política extrema
- **Foco**: Democracia e economia
- **Temas principais**: Democracia (199.8), Economia (120.8), Armas/Segurança (40.0)

### 2022
- **Contexto**: Retorno de Lula, pandemia, fake news
- **Foco**: Democracia, saúde e economia
- **Temas principais**: Democracia (136.6), Economia (65.9), Saúde (53.8)

## 💡 Recomendações

### Para Candidatos
- Focar em temas de saúde e bem-estar (crescimento de 657%)
- Abordar questões de corrupção e transparência (crescimento de 2.659%)
- Preparar-se para debates sobre fake news e desinformação
- Considerar que temas econômicos tradicionais têm menor apelo

### Para Pesquisadores
- Investigar a relação entre polarização e temas identitários
- Analisar o impacto da pandemia na agenda política
- Estudar a evolução do discurso sobre corrupção
- Examinar a mudança de foco de temas internacionais para domésticos

### Para Cidadãos
- Estar atentos ao uso de fake news em campanhas
- Demandar propostas concretas sobre saúde e bem-estar
- Questionar candidatos sobre transparência e combate à corrupção
- Não se deixar levar por polarizações baseadas apenas em valores

## 🛠️ Como Usar os Scripts

### 1. Análise Temporal Completa
```bash
python3 temporal_analysis.py
```
**Gera**:
- Gráficos de evolução temporal
- Dados em CSV
- Relatório de tendências

### 2. Resumo Executivo
```bash
python3 summary_temporal_analysis.py
```
**Gera**:
- Análise detalhada das tendências
- Contexto histórico
- Padrões sociopolíticos
- Recomendações

## 📁 Estrutura de Arquivos

```
transcripts/
├── temporal_analysis.py              # Script principal de análise
├── summary_temporal_analysis.py      # Script de resumo executivo
├── analise_temporal/                 # Resultados da análise
│   ├── evolucao_temas_temporais.png  # Gráfico de evolução
│   ├── heatmap_temas_por_ano.png     # Heatmap de temas
│   ├── comparacao_temas_por_ano.png  # Comparação entre anos
│   ├── frequencia_temas_por_ano.csv  # Dados de frequência
│   ├── contagem_temas_por_ano.csv    # Dados de contagem
│   ├── analise_tendencias_temas.csv  # Análise de tendências
│   └── relatorio_tendencias.txt      # Relatório detalhado
└── README_TEMPORAL_ANALYSIS.md       # Este arquivo
```

## 📊 Visualizações Geradas

### 1. Evolução Temporal dos Principais Temas
- Linha do tempo mostrando a evolução dos 8 temas mais frequentes
- Permite identificar tendências de crescimento e declínio

### 2. Heatmap de Temas por Ano
- Matriz colorida mostrando a intensidade de cada tema por ano
- Facilita a identificação de padrões e mudanças

### 3. Comparação entre Anos
- Gráfico de barras comparando temas entre diferentes anos
- Mostra mudanças relativas entre períodos

## 🔬 Limitações e Considerações

### Limitações Metodológicas
- **Amostra limitada**: Apenas 4 anos com debates disponíveis
- **Diferentes formatos**: Debates de diferentes emissoras podem ter formatos distintos
- **Dicionário temático**: Baseado em palavras-chave, pode não capturar nuances

### Considerações para Análise
- **Contexto histórico**: Mudanças podem refletir eventos específicos (pandemia, escândalos)
- **Formato dos debates**: Diferentes estruturas podem influenciar a frequência de temas
- **Cobertura midiática**: Debates de diferentes emissoras podem ter focos distintos

## 🚀 Próximos Passos

### Melhorias Técnicas
- [ ] Incluir mais anos de debates
- [ ] Refinar o dicionário temático
- [ ] Implementar análise de sentimento
- [ ] Adicionar análise de co-ocorrência de temas

### Expansões Analíticas
- [ ] Análise por candidato
- [ ] Comparação entre emissoras
- [ ] Análise de turnos eleitorais
- [ ] Correlação com resultados eleitorais

## 📚 Referências

- **Metodologia**: Análise de conteúdo baseada em dicionários temáticos
- **Processamento**: Técnicas de NLP para normalização e contagem
- **Visualização**: Matplotlib e Seaborn para gráficos temporais
- **Análise**: Correlação temporal e análise de tendências

## 👥 Contribuições

Este projeto foi desenvolvido para análise acadêmica e pode ser expandido com:
- Novos dados de debates
- Melhorias no dicionário temático
- Análises adicionais
- Visualizações interativas

---

**Desenvolvido para análise temporal de debates presidenciais brasileiros (2002-2022)**
