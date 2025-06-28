import os
import pandas as pd
import unicodedata
import re
import matplotlib.pyplot as plt

# Função para normalizar textos
def normalize_text(text):
    if pd.isnull(text):
        return ""
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower()

# Termos religiosos normalizados
religious_terms = ['Pastor', 'Pastora', 'Padre','Missionário', 'Missionária', 'Irmão', 'Irmã', 'Reverendo', 'Reverenda', 'Bispo', 'Bispa', 'Apostolo', 'Apostola', 'de Madureira']
religious_terms_normalized = [normalize_text(term) for term in religious_terms]
regex_terms = '|'.join(religious_terms_normalized)

# Caminho para a pasta TSE_data
base_path = os.path.join('..','TSE_data')

# Lista para armazenar resultados por ano
results = []

# Percorre pastas na pasta TSE_data
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path) and folder.endswith('_elections'):
        ano = re.findall(r'\d{4}', folder)
        if not ano:
            continue
        ano = ano[0]

        file_path = os.path.join(folder_path, f'consulta_cand_{ano}_BRASIL.csv')
        if not os.path.exists(file_path):
            print(f'Arquivo não encontrado para {ano}')
            continue

        print(f'Processando {ano}...')
        df = pd.read_csv(file_path, sep=';', encoding='latin1')
        total_candidatos = len(df)

        df['NM_URNA_CANDIDATO_NORMALIZED'] = df['NM_URNA_CANDIDATO'].apply(normalize_text)
        religious_name_filter = df[df['NM_URNA_CANDIDATO_NORMALIZED'].str.contains(regex_terms, na=False)]
        total_religiosos = len(religious_name_filter)

        percent_religiosos = (total_religiosos / total_candidatos) * 100

        results.append({
            'Ano': int(ano),
            'Total Candidaturas': total_candidatos,
            'Candidatos com Nome Religioso': total_religiosos,
            '% Nome Religioso': round(percent_religiosos, 2)
        })

# Cria dataframe com resultados
if not results:
    print("Nenhum dado encontrado. Verifique as pastas e arquivos.")
else:
    result_df = pd.DataFrame(results).sort_values(by='Ano')
    result_df.to_csv('percentual_nomes_religiosos_por_ano.csv', index=False)
    print(result_df)

    # Classificação
    municipais = [1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]
    result_df['Tipo Eleição'] = result_df['Ano'].apply(lambda x: 'Municipal' if x in municipais else 'Federal')

    df_municipal = result_df[result_df['Tipo Eleição'] == 'Municipal']
    df_federal = result_df[result_df['Tipo Eleição'] == 'Federal']

    # Gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(df_municipal['Ano'], df_municipal['% Nome Religioso'], marker='o', color='#fb6a3d', label='Eleições Municipais')
    plt.plot(df_federal['Ano'], df_federal['% Nome Religioso'],  marker='o', color='#361d20', label='Eleições Federais')

    for _, row in result_df.iterrows():
        plt.text(row['Ano'], row['% Nome Religioso'] + 0.1, f"{row['% Nome Religioso']}%", 
                 ha='center', va='top', fontsize=8)

    plt.title('% de Candidaturas com Nome Religioso por Tipo de Eleição')
    plt.xlabel('Ano')
    plt.ylabel('% Candidaturas com nome de urna religioso')
    plt.grid(False)
    plt.xticks(result_df['Ano'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('grafico_candidaturas_religiosas.png')
    plt.show()
