#!/usr/bin/env python3
"""
Análise temporal corrigida que considera adequadamente o denominador (quantidade de debates por ano)
"""

import os
import re
import logging
import unicodedata
from collections import defaultdict
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# === CONFIGURAÇÃO DO LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

# === DICIONÁRIO DE TEMAS (mesmo do arquivo original) ===
temas_padroes = {
    "educacao": {
        "frases": [
            r"ensino medio", r"ensino fundamental", r"educacao infantil", r"educacao basica",
            r"universidade", r"escola", r"professor", r"aluno", r"creche", r"merenda",
        ],
        "palavras": [
            r"educa\w*", r"escola\w*", r"ensino\w*", r"universidad\w*", r"professor\w*", 
            r"alun\w*", r"creche\w*", r"merend\w*", r"enem\b", r"prouni\b", r"fies\b",
            r"pronatec\w*", r"escola\w* tecnica", r"formacao\w*", r"qualificacao\w*",
        ],
    },
    "economia": {
        "frases": [r"reforma tributaria", r"plano real", r"inflacao", r"crescimento economico"],
        "palavras": [
            r"econom\w*", r"inflac\w*", r"empreg\w*", r"desempreg\w*", r"trabalh\w*", r"salari\w*",
            r"impost\w*", r"rend\w*", r"pib\b", r"crescimento\w*", r"juros\w*", r"cambio\w*",
            r"dolar\w*", r"real\b", r"dolar\b", r"mercado\w*", r"investiment\w*",
        ],
    },
    "armas_seguranca": {
        "frases": [r"seguranca publica", r"porte de arma\w*", r"posse de arma\w*", r"violencia"],
        "palavras": [
            r"arma\w*", r"desarmament\w*", r"polici\w*", r"homicid\w*", r"violenc\w*", r"seguranc\w*",
            r"crime\w*", r"criminal\w*", r"assassinat\w*", r"trafico\w*", r"drog\w*",
            r"policia\w* federal", r"policia\w* militar", r"forca\w* armada\w*",
        ],
    },
    "religiao": {
        "frases": [r"liberdade de culto", r"moral crist\w*", r"gracas a deus"],
        "palavras": [
            r"relig\w*", r"igreja\w*", r"deus\b", r"evangelic\w*", r"catolic\w*", r"pastor\w*", 
            r"biblia\w*", r"culto\w*", r"fe\b", r"oracao\b", r"padre", r"abenc\w", r"crist\b",
        ],
    },
    "moralidade_sexualidade": {
        "frases": [r"bons costumes", r"educacao sexual", r"familia tradicional", r"ideologia de genero"],
        "palavras": [
            r"moral\w*", r"famil\w*", r"valore\w*", r"costum\w*", r"tradic\w*", r"conservador\w*",
            r"sexual\w*", r"genero\w*", r"lgbt\w*", r"homossex\w*", r"transex\w*", r"aborto\w*", 
            r"sexo\b", r"drog\b", r"casamento\w*", r"divorcio\w*",
        ],
    },
    "democracia": {
        "frases": [r"liberdade de expressao", r"estado de direito", r"democracia"],
        "palavras": [
            r"democrac\w*", r"constituic\w*", r"stf\b", r"eleic\w*", r"urna\w*", r"golpe\w*", 
            r"ditadur\w*", r"instituic\w*", r"liberdade\w*", r"voto\w*", r"candidat\w*",
            r"partido\w*", r"politic\w*", r"governo\w*", r"presidente\w*",
        ],
    },
    "corrupcao": {
        "frases": [r"lava jato", r"cassacao do mandato", r"crime de responsabilidade", r"mensalao"],
        "palavras": [
            r"corrupc\w*", r"mensalao\b", r"rachadin\w*", r"propin\w*", r"desvi\w*", 
            r"lavagem de dinheiro", r"corrup\w*", r"improbidad\w*", r"peculato\b", 
            r"enriquecimento ilic\w*", r"laranj\w*", r"petrolao\b", r"delacao\w*",
        ],
    },
    "fake_news": {
        "frases": [
            r"fake news", r"noticia falsa", r"noticias falsas", r"checagem de fatos", 
            r"verificacao de fatos", r"fact checking",
        ],
        "palavras": [
            r"desinformacao\w*", r"misinformacao\w*", r"boato\w*", r"mentir\w*", r"mentira\w*", 
            r"engan\w*", r"fals\w*", r"propaganda\w*", r"calunia\w*", r"difamacao\w*",
        ],
    },
    "saude": {
        "frases": [
            r"saude publica", r"sistema unico de saude", r"atencao primaria a saude",
            r"passaporte vacinal", r"obrigatoriedade da vacina", r"unidade de terapia intensiva",
        ],
        "palavras": [
            r"saude\b", r"doenca\w*", r"pandemi\w*", r"covid\w*", r"coronavirus\w*", r"sars\w*cov\w*",
            r"vacina\w*", r"vacin\w*", r"imuniz\w*", r"dose\w*", r"sus\b", r"hospita\w*", 
            r"leito\w*", r"uti\b", r"oxigen\w*", r"respirad\w*", r"mascar\w*", r"quarenten\w*", 
            r"isolament\w*", r"lockdown\b", r"epidem\w*", r"contagi\w*", r"transmiss\w*",
            r"medic\w*", r"remedio\w*", r"tratament\w*",
        ],
    },
    "previdencia": {
        "frases": [r"previdencia social", r"aposentadoria", r"inss"],
        "palavras": [
            r"previdenc\w*", r"aposentador\w*", r"inss\b", r"aposentad\w*", r"pensionist\w*",
            r"fator previdenciario", r"reforma da previdencia", r"capitalizacao\w*",
        ],
    },
    "infraestrutura": {
        "frases": [r"obras publicas", r"infraestrutura", r"transporte publico"],
        "palavras": [
            r"infraestrutur\w*", r"obra\w*", r"construcao\w*", r"estrada\w*", r"rodovia\w*",
            r"ferrovia\w*", r"porto\w*", r"aeroporto\w*", r"energia\w*", r"agua\w*",
            r"saneament\w*", r"transporte\w*", r"metro\w*", r"onibu\w*",
        ],
    },
    "agricultura": {
        "frases": [r"agricultura familiar", r"agronegocio", r"reforma agraria"],
        "palavras": [
            r"agricultur\w*", r"agronegoci\w*", r"fazenda\w*", r"rural\w*", r"campo\w*",
            r"produtor\w* rural", r"agricultor\w*", r"reforma agraria", r"terra\w*",
            r"credito rural", r"financiamento\w* agricol\w*",
        ],
    },
    "meio_ambiente": {
        "frases": [r"meio ambiente", r"desenvolvimento sustentavel", r"mudanca climatica"],
        "palavras": [
            r"ambiente\w*", r"ecolog\w*", r"sustentavel\w*", r"clima\w*", r"poluicao\w*",
            r"floresta\w*", r"amazonia\w*", r"preservacao\w*", r"reciclagem\w*",
            r"energia renovavel", r"energia solar", r"energia eolica",
        ],
    },
    "internacional": {
        "frases": [r"relacoes internacionais", r"politica externa", r"mercosul"],
        "palavras": [
            r"internacional\w*", r"exterior\w*", r"mercosul\b", r"alca\b", r"brics\b",
            r"onu\b", r"fmi\b", r"banco mundial", r"exportacao\w*", r"importacao\w*",
            r"comercio exterior", r"diplomacia\w*",
        ],
    },
}

def extrair_ano_do_nome_arquivo(nome_arquivo):
    """Extrai o ano do nome do arquivo de debate."""
    # Padrão específico para anos entre sublinhados (ex: _2002_, _2014_, _2018_, _2022_)
    padrao = r'_(\d{4})_'
    
    match = re.search(padrao, nome_arquivo)
    if match:
        return int(match.group(1))
    
    return None

def normalizar_texto(texto):
    """Normaliza o texto removendo acentos e convertendo para minúsculas."""
    texto_lower = texto.lower()
    texto_norm, _ = normalizar_com_mapa(texto_lower)
    return texto_norm

def normalizar_com_mapa(texto_base):
    """Normaliza removendo acentos (ASCII) e retorna o texto normalizado."""
    norm_chars = []
    for ch in texto_base:
        decomposed = unicodedata.normalize("NFD", ch)
        # remove marcas de acento
        filtered = ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn')
        # mantém somente ASCII imprimível
        try:
            filtered_ascii = filtered.encode('ascii', 'ignore').decode('ascii')
        except Exception:
            filtered_ascii = ''
        if filtered_ascii:
            norm_chars.append(filtered_ascii)
    return ''.join(norm_chars), []

def contar_tema(texto_base, padroes):
    """Conta ocorrências de um tema no texto."""
    restante = texto_base
    total = 0
    
    # Frases primeiro (evita dupla contagem)
    for fr in padroes.get("frases", []):
        matches = list(re.finditer(rf"\b{fr}\b", restante))
        total += len(matches)
        if matches:
            restante = re.sub(rf"\b{fr}\b", " ", restante)
    
    # Palavras
    for pw in padroes.get("palavras", []):
        total += len(re.findall(rf"\b{pw}\b", restante))
    
    return total

def analisar_debate(arquivo_path):
    """Analisa um arquivo de debate e retorna contagem de temas."""
    try:
        with open(arquivo_path, "r", encoding="utf-8") as f:
            texto = f.read()
        
        # Normaliza o texto
        texto_norm = normalizar_texto(texto)
        
        # Conta temas
        temas_contagem = {}
        for tema, padroes in temas_padroes.items():
            temas_contagem[tema] = contar_tema(texto_norm, padroes)
        
        # Calcula frequência por 10k palavras
        palavras_totais = max(1, len(re.findall(r"\b\w+\b", texto_norm)))
        temas_freq = {}
        for tema, contagem in temas_contagem.items():
            temas_freq[tema] = (contagem / palavras_totais) * 10000
        
        return {
            'arquivo': os.path.basename(arquivo_path),
            'tamanho_texto': len(texto),
            'palavras_totais': palavras_totais,
            'temas_contagem': temas_contagem,
            'temas_freq': temas_freq
        }
        
    except Exception as e:
        logging.error(f"Erro ao analisar {arquivo_path}: {e}")
        return None

def encontrar_arquivos_debate(diretorio):
    """Encontra todos os arquivos de debate no diretório."""
    arquivos_debate = []
    
    for arquivo in os.listdir(diretorio):
        # Inclui tanto "debate" quanto "dabate" (erro de digitação)
        if (arquivo.endswith('.txt') and 
            ('debate' in arquivo.lower() or 'dabate' in arquivo.lower()) and
            not arquivo.startswith('lda_') and
            not arquivo.startswith('temas_') and
            not arquivo.startswith('summary_') and
            not arquivo.startswith('temporal_') and
            not arquivo.startswith('themes.') and
            not arquivo.startswith('transcript.') and
            not arquivo.startswith('run.')):
            ano = extrair_ano_do_nome_arquivo(arquivo)
            if ano:
                arquivos_debate.append({
                    'arquivo': arquivo,
                    'caminho': os.path.join(diretorio, arquivo),
                    'ano': ano
                })
    
    return sorted(arquivos_debate, key=lambda x: x['ano'])

def agrupar_por_ano_corrigido(analises):
    """Agrupa análises por ano e calcula frequências ponderadas corretamente."""
    anos = defaultdict(list)
    
    for analise in analises:
        if analise:
            ano = extrair_ano_do_nome_arquivo(analise['arquivo'])
            if ano:
                anos[ano].append(analise)
    
    # Calcula frequências ponderadas por ano
    resultados_ano = {}
    for ano, debates in anos.items():
        if debates:
            # Soma total de ocorrências e palavras por ano
            temas_contagem_total = defaultdict(int)
            palavras_totais_ano = 0
            
            for debate in debates:
                palavras_totais_ano += debate['palavras_totais']
                for tema, contagem in debate['temas_contagem'].items():
                    temas_contagem_total[tema] += contagem
            
            # Calcula frequência ponderada por 10k palavras
            temas_freq_ponderada = {}
            for tema, contagem_total in temas_contagem_total.items():
                if palavras_totais_ano > 0:
                    temas_freq_ponderada[tema] = (contagem_total / palavras_totais_ano) * 10000
                else:
                    temas_freq_ponderada[tema] = 0
            
            resultados_ano[ano] = {
                'debates': len(debates),
                'palavras_totais': palavras_totais_ano,
                'temas_contagem_total': dict(temas_contagem_total),
                'temas_freq_ponderada': temas_freq_ponderada
            }
    
    return resultados_ano

def criar_graficos_temporais_corrigidos(resultados_ano, output_dir):
    """Cria gráficos temporais dos temas com frequências corrigidas."""
    
    # Prepara dados para gráficos
    anos = sorted(resultados_ano.keys())
    temas = list(temas_padroes.keys())
    
    # Gráfico 1: Evolução da frequência dos principais temas
    plt.figure(figsize=(15, 10))
    
    # Seleciona os temas mais frequentes, garantindo que temas de interesse da tese sejam incluídos
    freq_total = defaultdict(float)
    for ano_data in resultados_ano.values():
        for tema, freq in ano_data['temas_freq_ponderada'].items():
            freq_total[tema] += freq
    
    # Temas de interesse da tese (sempre incluídos)
    temas_tese = {'religiao', 'moralidade_sexualidade'}
    
    # Seleciona os 6 temas mais frequentes (além dos temas da tese)
    temas_ordenados = sorted(freq_total.items(), key=lambda x: x[1], reverse=True)
    temas_principais = []
    
    # Adiciona temas da tese primeiro
    for tema in temas_tese:
        if tema in freq_total:
            temas_principais.append(tema)
    
    # Adiciona os 6 temas mais frequentes (excluindo os já incluídos)
    for tema, _ in temas_ordenados:
        if tema not in temas_principais and len(temas_principais) < 8:
            temas_principais.append(tema)
    
    # Cores específicas para temas de interesse da tese
    cores_temas = {
        'religiao': '#8B0000',  # Vermelho escuro
        'moralidade_sexualidade': '#DC143C',  # Vermelho carmesim
    }
    
    for i, tema in enumerate(temas_principais):
        valores = [resultados_ano[ano]['temas_freq_ponderada'].get(tema, 0) for ano in anos]
        
        # Define cor e estilo baseado no tema
        if tema in cores_temas:
            cor = cores_temas[tema]
            linewidth = 3
            markersize = 8
            alpha = 1.0
        else:
            cor = '#CCCCCC'  # Cinza claro
            linewidth = 1.5
            markersize = 4
            alpha = 0.7
        
        plt.plot(anos, valores, marker='o', linewidth=linewidth, markersize=markersize, 
                color=cor, alpha=alpha, label=tema.replace('_', ' ').title())
    
    plt.title('Evolução Temporal dos Principais Temas nos Debates Presidenciais', fontsize=16, fontweight='bold')
    plt.xlabel('Ano das Eleições', fontsize=12)
    plt.ylabel('Frequência por 10.000 palavras (ponderada)', fontsize=12)
    plt.xticks(anos, [f'{ano}' for ano in anos], rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'evolucao_temas_temporais_corrigido.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 2: Heatmap de temas por ano
    plt.figure(figsize=(16, 10))
    
    # Matriz de dados para o heatmap
    matriz_dados = []
    for tema in temas:
        linha = [resultados_ano[ano]['temas_freq_ponderada'].get(tema, 0) for ano in anos]
        matriz_dados.append(linha)
    
    matriz_dados = np.array(matriz_dados)
    
    # Cria heatmap
    sns.heatmap(matriz_dados, 
                xticklabels=anos, 
                yticklabels=[tema.replace('_', ' ').title() for tema in temas],
                annot=True, 
                fmt='.1f', 
                cmap='YlOrRd',
                cbar_kws={'label': 'Frequência por 10.000 palavras (ponderada)'})
    
    plt.title('Heatmap de Temas por Ano - Debates Presidenciais', fontsize=16, fontweight='bold')
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Tema', fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'heatmap_temas_por_ano_corrigido.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 3: Comparação entre anos (barras)
    plt.figure(figsize=(15, 8))
    
    # Seleciona os 6 temas mais relevantes
    temas_relevantes = temas_principais[:6]
    
    x = np.arange(len(anos))
    width = 0.15
    
    # Cores específicas para temas de interesse da tese
    cores_temas = {
        'religiao': '#8B0000',  # Vermelho escuro
        'moralidade_sexualidade': '#DC143C',  # Vermelho carmesim
    }
    
    for i, tema in enumerate(temas_relevantes):
        valores = [resultados_ano[ano]['temas_freq_ponderada'].get(tema, 0) for ano in anos]
        
        # Define cor e estilo baseado no tema
        if tema in cores_temas:
            cor = cores_temas[tema]
            alpha = 1.0
        else:
            cor = '#CCCCCC'  # Cinza claro
            alpha = 0.7
        
        plt.bar(x + i * width, valores, width, label=tema.replace('_', ' ').title(), 
                color=cor, alpha=alpha)
    
    plt.title('Comparação de Temas por Ano - Debates Presidenciais', fontsize=16, fontweight='bold')
    plt.xlabel('Ano das Eleições', fontsize=12)
    plt.ylabel('Frequência por 10.000 palavras (ponderada)', fontsize=12)
    plt.xticks(x + width * 2.5, [f'{ano}' for ano in anos], rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'comparacao_temas_por_ano_corrigido.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def salvar_resultados_csv_corrigidos(resultados_ano, output_dir):
    """Salva resultados corrigidos em arquivos CSV."""
    
    # DataFrame com frequências ponderadas por ano
    dados_freq = []
    for ano, dados in resultados_ano.items():
        linha = {
            'ano': ano, 
            'debates': dados['debates'],
            'palavras_totais': dados['palavras_totais']
        }
        linha.update(dados['temas_freq_ponderada'])
        dados_freq.append(linha)
    
    df_freq = pd.DataFrame(dados_freq)
    output_path = os.path.join(output_dir, 'frequencia_temas_por_ano_corrigido.csv')
    df_freq.to_csv(output_path, index=False)
    
    # DataFrame com contagens totais por ano
    dados_cont = []
    for ano, dados in resultados_ano.items():
        linha = {
            'ano': ano, 
            'debates': dados['debates'],
            'palavras_totais': dados['palavras_totais']
        }
        linha.update(dados['temas_contagem_total'])
        dados_cont.append(linha)
    
    df_cont = pd.DataFrame(dados_cont)
    output_path = os.path.join(output_dir, 'contagem_temas_por_ano_corrigido.csv')
    df_cont.to_csv(output_path, index=False)
    
    return df_freq, df_cont

def main():
    """Função principal."""
    logging.info("Iniciando análise temporal CORRIGIDA de debates presidenciais...")
    
    # Diretório atual (onde estão os arquivos de debate)
    diretorio_atual = os.path.dirname(__file__)
    
    # Encontra arquivos de debate
    arquivos_debate = encontrar_arquivos_debate(diretorio_atual)
    logging.info(f"Encontrados {len(arquivos_debate)} arquivos de debate")
    
    for arquivo in arquivos_debate:
        logging.info(f"Arquivo: {arquivo['arquivo']} (Ano: {arquivo['ano']})")
    
    # Analisa cada debate
    analises = []
    for arquivo in arquivos_debate:
        logging.info(f"Analisando {arquivo['arquivo']}...")
        analise = analisar_debate(arquivo['caminho'])
        if analise:
            analises.append(analise)
    
    # Agrupa por ano (CORRIGIDO)
    resultados_ano = agrupar_por_ano_corrigido(analises)
    logging.info(f"Análise corrigida concluída para {len(resultados_ano)} anos")
    
    # Cria diretório de saída
    output_dir = os.path.join(diretorio_atual, 'analise_temporal_corrigida')
    os.makedirs(output_dir, exist_ok=True)
    
    # Cria gráficos corrigidos
    logging.info("Criando gráficos temporais corrigidos...")
    criar_graficos_temporais_corrigidos(resultados_ano, output_dir)
    
    # Salva resultados em CSV
    logging.info("Salvando resultados corrigidos em CSV...")
    df_freq, df_cont = salvar_resultados_csv_corrigidos(resultados_ano, output_dir)
    
    # Resumo final
    logging.info("=== RESUMO DA ANÁLISE CORRIGIDA ===")
    anos = sorted(resultados_ano.keys())
    logging.info(f"Período analisado: {min(anos)} - {max(anos)}")
    logging.info(f"Total de debates: {sum(resultados_ano[ano]['debates'] for ano in anos)}")
    
    # Mostra estatísticas por ano
    for ano in anos:
        dados = resultados_ano[ano]
        logging.info(f"Ano {ano}: {dados['debates']} debates, {dados['palavras_totais']:,} palavras")
    
    # Temas mais frequentes no período
    freq_total = defaultdict(float)
    for ano_data in resultados_ano.values():
        for tema, freq in ano_data['temas_freq_ponderada'].items():
            freq_total[tema] += freq
    
    temas_mais_frequentes = sorted(freq_total.items(), key=lambda x: x[1], reverse=True)[:5]
    logging.info("Temas mais frequentes no período (corrigido):")
    for tema, freq in temas_mais_frequentes:
        logging.info(f"  - {tema.replace('_', ' ').title()}: {freq:.1f}")
    
    logging.info(f"Análise temporal corrigida concluída! Resultados salvos em: {output_dir}")

if __name__ == "__main__":
    main()
