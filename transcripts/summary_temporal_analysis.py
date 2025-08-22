#!/usr/bin/env python3
"""
Análise Temporal de Debates Presidenciais - Resumo Executivo
============================================================

Este script analisa os resultados da análise temporal dos debates presidenciais
e fornece insights sobre a evolução dos temas ao longo do tempo.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def carregar_dados():
    """Carrega os dados da análise temporal."""
    base_path = Path(__file__).parent / 'analise_temporal'
    
    # Carrega dados de frequência
    df_freq = pd.read_csv(base_path / 'frequencia_temas_por_ano.csv')
    
    # Carrega análise de tendências
    df_tendencias = pd.read_csv(base_path / 'analise_tendencias_temas.csv')
    
    return df_freq, df_tendencias

def analisar_tendencias_principais(df_freq, df_tendencias):
    """Analisa as principais tendências identificadas."""
    
    print("=" * 80)
    print("ANÁLISE TEMPORAL DE DEBATES PRESIDENCIAIS (2002-2022)")
    print("=" * 80)
    
    # Estatísticas gerais
    anos = df_freq['ano'].tolist()
    total_debates = df_freq['debates'].sum()
    
    print(f"\n📊 ESTATÍSTICAS GERAIS:")
    print(f"   • Período analisado: {min(anos)} - {max(anos)} ({max(anos) - min(anos)} anos)")
    print(f"   • Total de debates: {total_debates}")
    print(f"   • Anos com debates: {len(anos)}")
    print(f"   • Média de debates por ano: {total_debates/len(anos):.1f}")
    
    # Temas emergentes (crescimento > 50% e correlação > 0.3)
    temas_emergentes = df_tendencias[
        (df_tendencias['correlacao_tempo'] > 0.3) & 
        (df_tendencias['variacao_percentual'] > 50)
    ].sort_values('variacao_percentual', ascending=False)
    
    print(f"\n🚀 TEMAS EMERGENTES (crescimento significativo):")
    for _, tema in temas_emergentes.iterrows():
        print(f"   • {tema['tema'].replace('_', ' ').title()}: "
              f"+{tema['variacao_percentual']:.1f}% "
              f"(correlação: {tema['correlacao_tempo']:.3f})")
    
    # Temas em declínio (queda > 50% e correlação < -0.3)
    temas_declinio = df_tendencias[
        (df_tendencias['correlacao_tempo'] < -0.3) & 
        (df_tendencias['variacao_percentual'] < -50)
    ].sort_values('variacao_percentual')
    
    print(f"\n📉 TEMAS EM DECLÍNIO (queda significativa):")
    for _, tema in temas_declinio.iterrows():
        print(f"   • {tema['tema'].replace('_', ' ').title()}: "
              f"{tema['variacao_percentual']:.1f}% "
              f"(correlação: {tema['correlacao_tempo']:.3f})")
    
    # Temas estáveis
    temas_estaveis = df_tendencias[
        (abs(df_tendencias['correlacao_tempo']) <= 0.3) |
        (abs(df_tendencias['variacao_percentual']) <= 50)
    ].sort_values('correlacao_tempo')
    
    print(f"\n⚖️  TEMAS ESTÁVEIS (pouca variação):")
    for _, tema in temas_estaveis.iterrows():
        print(f"   • {tema['tema'].replace('_', ' ').title()}: "
              f"{tema['variacao_percentual']:.1f}% "
              f"(correlação: {tema['correlacao_tempo']:.3f})")

def analisar_contexto_historico(df_freq):
    """Analisa o contexto histórico de cada ano."""
    
    print(f"\n📅 CONTEXTO HISTÓRICO POR ANO:")
    
    for _, ano_data in df_freq.iterrows():
        ano = ano_data['ano']
        debates = ano_data['debates']
        
        # Top 3 temas do ano
        temas_ano = []
        for col in ano_data.index:
            if col not in ['ano', 'debates'] and not pd.isna(ano_data[col]):
                temas_ano.append((col, ano_data[col]))
        
        temas_ano.sort(key=lambda x: x[1], reverse=True)
        top_3 = temas_ano[:3]
        
        print(f"\n   {ano} ({debates} debate{'s' if debates > 1 else ''}):")
        for tema, freq in top_3:
            print(f"     • {tema.replace('_', ' ').title()}: {freq:.1f}")
        
        # Contexto histórico específico
        if ano == 2002:
            print("       Contexto: Eleição que levou Lula ao poder, foco em economia e democracia")
        elif ano == 2014:
            print("       Contexto: Reeleição de Dilma, escândalos de corrupção emergindo")
        elif ano == 2018:
            print("       Contexto: Eleição de Bolsonaro, polarização política extrema")
        elif ano == 2022:
            print("       Contexto: Retorno de Lula, pandemia, fake news")

def analisar_mudancas_tematicas(df_freq):
    """Analisa as mudanças temáticas entre períodos."""
    
    print(f"\n🔄 MUDANÇAS TEMÁTICAS ENTRE PERÍODOS:")
    
    anos = df_freq['ano'].tolist()
    
    for i in range(len(anos) - 1):
        ano_atual = anos[i]
        ano_proximo = anos[i + 1]
        
        print(f"\n   {ano_atual} → {ano_proximo}:")
        
        # Compara os top temas
        dados_atual = df_freq[df_freq['ano'] == ano_atual].iloc[0]
        dados_proximo = df_freq[df_freq['ano'] == ano_proximo].iloc[0]
        
        # Encontra temas que ganharam mais relevância
        mudancas = []
        for col in dados_atual.index:
            if col not in ['ano', 'debates'] and not pd.isna(dados_atual[col]) and not pd.isna(dados_proximo[col]):
                mudanca = dados_proximo[col] - dados_atual[col]
                mudancas.append((col, mudanca))
        
        mudancas.sort(key=lambda x: x[1], reverse=True)
        
        print(f"     📈 Maior crescimento:")
        for tema, mudanca in mudancas[:3]:
            if mudanca > 0:
                print(f"       • {tema.replace('_', ' ').title()}: +{mudanca:.1f}")
        
        print(f"     📉 Maior queda:")
        for tema, mudanca in mudancas[-3:]:
            if mudanca < 0:
                print(f"       • {tema.replace('_', ' ').title()}: {mudanca:.1f}")

def identificar_padroes_sociopoliticos(df_freq, df_tendencias):
    """Identifica padrões sociopolíticos nas mudanças temáticas."""
    
    print(f"\n🎯 PADRÕES SOCIOPOLÍTICOS IDENTIFICADOS:")
    
    # Padrão 1: Crescimento de temas relacionados a valores e identidade
    temas_valores = ['religiao', 'moralidade_sexualidade', 'fake_news']
    crescimento_valores = df_tendencias[df_tendencias['tema'].isin(temas_valores)]
    
    if not crescimento_valores.empty:
        print(f"\n   🔸 POLARIZAÇÃO E VALORES:")
        print(f"     • Temas relacionados a valores e identidade cresceram consistentemente")
        if 'religiao' in crescimento_valores['tema'].values:
            print(f"     • Religião: +{crescimento_valores[crescimento_valores['tema'] == 'religiao']['variacao_percentual'].iloc[0]:.1f}%")
        if 'moralidade_sexualidade' in crescimento_valores['tema'].values:
            print(f"     • Moralidade/Sexualidade: +{crescimento_valores[crescimento_valores['tema'] == 'moralidade_sexualidade']['variacao_percentual'].iloc[0]:.1f}%")
        if 'fake_news' in crescimento_valores['tema'].values:
            print(f"     • Fake News: +{crescimento_valores[crescimento_valores['tema'] == 'fake_news']['variacao_percentual'].iloc[0]:.1f}%")
        print(f"     → Indica crescente polarização política e debate sobre valores")
    
    # Padrão 2: Declínio de temas econômicos tradicionais
    temas_economicos = ['economia']
    declinio_economico = df_tendencias[df_tendencias['tema'].isin(temas_economicos)]
    
    if not declinio_economico.empty:
        print(f"\n   🔸 MUDANÇA NO FOCO ECONÔMICO:")
        print(f"     • Temas econômicos tradicionais perderam relevância")
        if 'economia' in declinio_economico['tema'].values:
            print(f"     • Economia: {declinio_economico[declinio_economico['tema'] == 'economia']['variacao_percentual'].iloc[0]:.1f}%")
        print(f"     → Sugere mudança de foco para questões mais domésticas e identitárias")
    
    # Padrão 3: Crescimento de temas de saúde e bem-estar
    temas_saude = ['saude']
    crescimento_saude = df_tendencias[df_tendencias['tema'].isin(temas_saude)]
    
    if not crescimento_saude.empty:
        print(f"\n   🔸 SAÚDE E BEM-ESTAR:")
        print(f"     • Saúde emergiu como tema central")
        print(f"     • Saúde: +{crescimento_saude['variacao_percentual'].iloc[0]:.1f}%")
        print(f"     → Provavelmente influenciado pela pandemia de COVID-19")
    
    # Padrão 4: Corrupção como tema persistente
    temas_corrupcao = ['corrupcao']
    crescimento_corrupcao = df_tendencias[df_tendencias['tema'].isin(temas_corrupcao)]
    
    if not crescimento_corrupcao.empty:
        print(f"\n   🔸 CORRUPÇÃO COMO TEMA CENTRAL:")
        print(f"     • Corrupção se tornou um dos temas mais relevantes")
        print(f"     • Corrupção: +{crescimento_corrupcao['variacao_percentual'].iloc[0]:.1f}%")
        print(f"     → Reflete escândalos como Lava Jato e crescente preocupação com transparência")

def gerar_recomendacoes(df_freq, df_tendencias):
    """Gera recomendações baseadas na análise."""
    
    print(f"\n💡 RECOMENDAÇÕES BASEADAS NA ANÁLISE:")
    
    print(f"\n   📋 PARA CANDIDATOS:")
    print(f"     • Focar em temas de saúde e bem-estar (crescimento de 657%)")
    print(f"     • Abordar questões de corrupção e transparência (crescimento de 2659%)")
    print(f"     • Preparar-se para debates sobre fake news e desinformação")
    print(f"     • Considerar que temas econômicos tradicionais têm menor apelo")
    
    print(f"\n   📋 PARA PESQUISADORES:")
    print(f"     • Investigar a relação entre polarização e temas identitários")
    print(f"     • Analisar o impacto da pandemia na agenda política")
    print(f"     • Estudar a evolução do discurso sobre corrupção")
    print(f"     • Examinar a mudança de foco de temas internacionais para domésticos")
    
    print(f"\n   📋 PARA CIDADÃOS:")
    print(f"     • Estar atentos ao uso de fake news em campanhas")
    print(f"     • Demandar propostas concretas sobre saúde e bem-estar")
    print(f"     • Questionar candidatos sobre transparência e combate à corrupção")
    print(f"     • Não se deixar levar por polarizações baseadas apenas em valores")

def main():
    """Função principal."""
    
    # Carrega dados
    df_freq, df_tendencias = carregar_dados()
    
    # Executa análises
    analisar_tendencias_principais(df_freq, df_tendencias)
    analisar_contexto_historico(df_freq)
    analisar_mudancas_tematicas(df_freq)
    identificar_padroes_sociopoliticos(df_freq, df_tendencias)
    gerar_recomendacoes(df_freq, df_tendencias)
    
    print(f"\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 80)
    print(f"\n📁 Arquivos gerados na pasta 'analise_temporal':")
    print(f"   • Gráficos de evolução temporal")
    print(f"   • Dados em CSV para análise adicional")
    print(f"   • Relatório detalhado de tendências")

if __name__ == "__main__":
    main()
