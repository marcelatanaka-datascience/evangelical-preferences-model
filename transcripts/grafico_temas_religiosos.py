#!/usr/bin/env python3
"""
Gráfico específico dos temas religiosos e de moralidade
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def teste_significancia_tendencia(anos, valores, nome_tema):
    """Testa se a tendência temporal é estatisticamente significativa"""
    
    # Teste de correlação de Spearman (não paramétrico)
    spearman_corr, spearman_p = stats.spearmanr(anos, valores)
    
    # Teste de correlação de Pearson
    pearson_corr, pearson_p = stats.pearsonr(anos, valores)
    
    # Teste t para a correlação
    n = len(anos)
    t_stat = pearson_corr * np.sqrt((n - 2) / (1 - pearson_corr**2))
    t_p = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    
    # Teste de Mann-Kendall para tendência
    def mann_kendall_test(x, y):
        n = len(x)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(y[j] - y[i])
        
        var_s = n * (n - 1) * (2 * n + 5) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return s, z, p_value
    
    mk_s, mk_z, mk_p = mann_kendall_test(anos, valores)
    
    return {
        'tema': nome_tema,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        't_stat': t_stat,
        't_p': t_p,
        'mann_kendall_s': mk_s,
        'mann_kendall_z': mk_z,
        'mann_kendall_p': mk_p
    }

def criar_grafico_temas_religiosos():
    """Cria gráfico específico para os temas religiosos e de moralidade"""
    
    # Carrega dados
    df = pd.read_csv('analise_temporal_corrigida/frequencia_temas_por_ano_corrigido.csv')
    
    # Extrai dados dos temas de interesse
    anos = df['ano'].tolist()
    religiao = df['religiao'].tolist()
    moralidade = df['moralidade_sexualidade'].tolist()
    
    plt.figure(figsize=(12, 8))
    
    # Gráfico principal
    plt.plot(anos, religiao, marker='o', linewidth=3, markersize=8, 
             color='#8B0000', label='Religião', alpha=0.9)
    plt.plot(anos, moralidade, marker='s', linewidth=3, markersize=8, 
             color='#DC143C', label='Moralidade/Sexualidade', alpha=0.9)
    
    # Adiciona área sombreada
    plt.fill_between(anos, religiao, alpha=0.2, color='#8B0000')
    plt.fill_between(anos, moralidade, alpha=0.2, color='#DC143C')
    
    plt.title('Evolução dos Temas Religiosos e de Moralidade nos Debates Presidenciais\n(1989-2022)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Frequência por 10.000 palavras (ponderada)', fontsize=12)
    plt.xticks(anos, [f'{ano}' for ano in anos], rotation=0)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Adiciona anotações para pontos importantes
    max_religiao_idx = np.argmax(religiao)
    max_moralidade_idx = np.argmax(moralidade)
    
    plt.annotate(f'Máximo: {religiao[max_religiao_idx]:.1f}', 
                xy=(anos[max_religiao_idx], religiao[max_religiao_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#8B0000', alpha=0.7),
                fontsize=10, color='white')
    
    plt.annotate(f'Máximo: {moralidade[max_moralidade_idx]:.1f}', 
                xy=(anos[max_moralidade_idx], moralidade[max_moralidade_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#DC143C', alpha=0.7),
                fontsize=10, color='white')
    
    plt.tight_layout()
    plt.savefig('temas_religiosos_evolucao.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfico dos temas religiosos salvo em: temas_religiosos_evolucao.png")
    
    # Mostra estatísticas
    print(f"\n📊 ESTATÍSTICAS DOS TEMAS RELIGIOSOS:")
    print(f"{'Ano':<6} {'Religião':<12} {'Moralidade/Sexualidade':<25}")
    print("-" * 50)
    
    for i, ano in enumerate(anos):
        print(f"{ano:<6} {religiao[i]:<12.1f} {moralidade[i]:<25.1f}")
    
    # Análise de tendências com testes de significância
    print(f"\n📈 ANÁLISE DE TENDÊNCIAS E SIGNIFICÂNCIA ESTATÍSTICA:")
    print("="*80)
    
    # Testa religião
    resultado_religiao = teste_significancia_tendencia(anos, religiao, "Religião")
    print(f"\n🔴 RELIGIÃO:")
    print(f"  Correlação de Pearson: {resultado_religiao['pearson_corr']:.3f} (p = {resultado_religiao['pearson_p']:.3f})")
    print(f"  Correlação de Spearman: {resultado_religiao['spearman_corr']:.3f} (p = {resultado_religiao['spearman_p']:.3f})")
    print(f"  Teste Mann-Kendall: S = {resultado_religiao['mann_kendall_s']:.0f}, Z = {resultado_religiao['mann_kendall_z']:.3f} (p = {resultado_religiao['mann_kendall_p']:.3f})")
    
    # Interpretação da significância
    alpha = 0.05
    religiao_significativo = resultado_religiao['pearson_p'] < alpha
    religiao_tendencia = "Crescente" if resultado_religiao['pearson_corr'] > 0 else "Decrescente" if resultado_religiao['pearson_corr'] < 0 else "Estável"
    print(f"  Tendência: {religiao_tendencia}")
    print(f"  Significativo (p < 0.05): {'✅ SIM' if religiao_significativo else '❌ NÃO'}")
    
    # Testa moralidade
    resultado_moralidade = teste_significancia_tendencia(anos, moralidade, "Moralidade/Sexualidade")
    print(f"\n🔴 MORALIDADE/SEXUALIDADE:")
    print(f"  Correlação de Pearson: {resultado_moralidade['pearson_corr']:.3f} (p = {resultado_moralidade['pearson_p']:.3f})")
    print(f"  Correlação de Spearman: {resultado_moralidade['spearman_corr']:.3f} (p = {resultado_moralidade['spearman_p']:.3f})")
    print(f"  Teste Mann-Kendall: S = {resultado_moralidade['mann_kendall_s']:.0f}, Z = {resultado_moralidade['mann_kendall_z']:.3f} (p = {resultado_moralidade['mann_kendall_p']:.3f})")
    
    # Interpretação da significância
    moralidade_significativo = resultado_moralidade['pearson_p'] < alpha
    moralidade_tendencia = "Crescente" if resultado_moralidade['pearson_corr'] > 0 else "Decrescente" if resultado_moralidade['pearson_corr'] < 0 else "Estável"
    print(f"  Tendência: {moralidade_tendencia}")
    print(f"  Significativo (p < 0.05): {'✅ SIM' if moralidade_significativo else '❌ NÃO'}")
    
    print(f"\n💡 INTERPRETAÇÃO:")
    print(f"  - p < 0.05: Tendência estatisticamente significativa")
    print(f"  - p ≥ 0.05: Tendência não é estatisticamente significativa")
    print(f"  - Correlação positiva: Crescimento ao longo do tempo")
    print(f"  - Correlação negativa: Declínio ao longo do tempo")

if __name__ == "__main__":
    criar_grafico_temas_religiosos()
