"""
Script para criar variável dummy_ethnic a partir das variáveis de etnia de cada onda.
Usa leitura direta de CSV para evitar conflitos com indexes do banco.

Passo 1: Identifica as variáveis de etnia em cada onda
Passo 2: Converte valores negativos em missing
Passo 3: Recodifica em dummy_ethnic (76001 = 1, outros = 0)
"""

import csv
import pandas as pd
from pathlib import Path
from collections import Counter

# Configuração base
BASE_DIR = Path(__file__).parent

# Configuração de cada onda
WAVES_CONFIG = {
    'wave-2-1991-br': {
        'file_original': 'WV2_Data_Brazil_Csv_v1.6.1.csv',
        'variable': 'V369',
        'year': 1991
    },
    'wave-3-1997-br': {
        'file_original': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'variable': 'V233',
        'year': 1997
    },
    'wave-5-2006-br': {
        'file_original': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'variable': 'V256',
        'year': 2005
    },
    'wave-6-2014-br': {
        'file_original': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'variable': 'V254',
        'year': 2014
    },
    'wave-7-2018-br': {
        'file_original': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'variable': 'Q290',
        'year': 2018
    }
}

def find_variable_index(header_row, variable_name):
    """Encontra o índice da variável no cabeçalho."""
    for i, col in enumerate(header_row):
        col_clean = col.strip('"').strip()
        if col_clean == variable_name:
            return i
    return None

def process_wave(wave_dir, config):
    """Processa uma onda completa."""
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {wave_dir.upper()}")
    print(f"{'='*70}")
    
    file_original = config['file_original']
    variable = config['variable']
    year = config['year']
    
    # Caminho do arquivo
    file_path = BASE_DIR / wave_dir / file_original
    
    if not file_path.exists():
        print(f"⚠️  Arquivo não encontrado: {file_path}")
        return None
    
    print(f"\n[PASSO 1] Carregando dados originais...")
    print(f"  Arquivo: {file_original}")
    print(f"  Variável: {variable}")
    print(f"  Ano: {year}")
    
    # Ler CSV manualmente
    rows = []
    header = None
    var_index = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';', quotechar='"')
            
            # Ler cabeçalho
            header = next(reader)
            var_index = find_variable_index(header, variable)
            
            if var_index is None:
                print(f"  ❌ Variável {variable} não encontrada no arquivo!")
                print(f"  Primeiras colunas: {header[:20]}")
                return None
            
            print(f"  ✓ Variável {variable} encontrada no índice {var_index}")
            
            # Ler todas as linhas
            for row_num, row in enumerate(reader, start=2):
                # Garantir que a linha tem o mesmo número de colunas do header
                if len(row) > len(header):
                    row = row[:len(header)]
                elif len(row) < len(header):
                    row.extend([''] * (len(header) - len(row)))
                rows.append(row)
        
        print(f"  ✓ Total de linhas lidas: {len(rows)}")
        
    except Exception as e:
        print(f"  ❌ Erro ao ler arquivo: {e}")
        return None
    
    # Passo 2: Processar valores - converter negativos em missing
    print(f"\n[PASSO 2] Processando valores (convertendo negativos em missing)...")
    
    # Adicionar coluna dummy_ethnic ao header
    header.append('dummy_ethnic')
    dummy_index = len(header) - 1
    
    # Estatísticas antes do processamento
    original_values = []
    for row in rows:
        if len(row) > var_index:
            value_str = row[var_index].strip('"').strip()
            try:
                if value_str:
                    value = float(value_str)
                    original_values.append(value)
            except (ValueError, TypeError):
                pass
    
    print(f"  Valores originais encontrados: {len(original_values)}")
    if original_values:
        freq_original = Counter(original_values)
        print(f"  Valores únicos: {sorted(freq_original.keys())[:10]}...")
        negative_count = sum(1 for v in original_values if v < 0)
        print(f"  Valores negativos: {negative_count}")
    
    # Processar cada linha
    processed_count = 0
    missing_count = 0
    negative_to_missing = 0
    dummy_1_count = 0
    dummy_0_count = 0
    
    for row in rows:
        # Garantir que a linha tem espaço para a nova coluna
        while len(row) < len(header):
            row.append('')
        
        if len(row) > var_index:
            value_str = row[var_index].strip('"').strip()
            
            try:
                if value_str:
                    value = float(value_str)
                    
                    # Passo 2: Se negativo, converter para missing
                    if value < 0:
                        value = None
                        negative_to_missing += 1
                    
                    # Passo 3: Recodificar em dummy_ethnic
                    if value is None:
                        dummy_value = ''
                        missing_count += 1
                    elif value == 76001.0:
                        dummy_value = '1'
                        dummy_1_count += 1
                    else:
                        dummy_value = '0'
                        dummy_0_count += 1
                    
                    row[dummy_index] = dummy_value
                    processed_count += 1
                else:
                    # Valor vazio
                    row[dummy_index] = ''
                    missing_count += 1
            except (ValueError, TypeError):
                # Valor não numérico
                row[dummy_index] = ''
                missing_count += 1
        else:
            # Linha sem a variável
            row.append('')
            missing_count += 1
    
    print(f"  ✓ Valores processados: {processed_count}")
    print(f"  ✓ Valores negativos convertidos para missing: {negative_to_missing}")
    print(f"  ✓ dummy_ethnic = 1 (76001): {dummy_1_count}")
    print(f"  ✓ dummy_ethnic = 0 (outros): {dummy_0_count}")
    print(f"  ✓ Missing: {missing_count}")
    
    # Salvar arquivo processado
    print(f"\n[PASSO 3] Salvando arquivo processado...")
    
    output_file = BASE_DIR / wave_dir / f"{file_original.replace('.csv', '')}_com_dummy_ethnic.csv"
    
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            # Escrever cabeçalho
            writer.writerow(header)
            
            # Escrever linhas
            for row in rows:
                writer.writerow(row)
        
        print(f"  ✓ Arquivo salvo: {output_file.name}")
        
        # Estatísticas finais
        print(f"\n[RESUMO]")
        print(f"  Total de linhas: {len(rows)}")
        print(f"  dummy_ethnic = 1: {dummy_1_count} ({dummy_1_count/len(rows)*100:.2f}%)")
        print(f"  dummy_ethnic = 0: {dummy_0_count} ({dummy_0_count/len(rows)*100:.2f}%)")
        print(f"  Missing: {missing_count} ({missing_count/len(rows)*100:.2f}%)")
        
        return {
            'wave': wave_dir,
            'year': year,
            'variable': variable,
            'total': len(rows),
            'dummy_1': dummy_1_count,
            'dummy_0': dummy_0_count,
            'missing': missing_count,
            'output_file': output_file.name
        }
        
    except Exception as e:
        print(f"  ❌ Erro ao salvar arquivo: {e}")
        return None

def main():
    """Função principal."""
    print("="*70)
    print("CRIAÇÃO DA VARIÁVEL dummy_ethnic")
    print("="*70)
    print("\nEste script processa as variáveis de etnia de cada onda:")
    print("  - Wave 2 (1991): V369")
    print("  - Wave 3 (1997): V233")
    print("  - Wave 5 (2005): V256")
    print("  - Wave 6 (2014): V254")
    print("  - Wave 7 (2018): Q290")
    print("\nRecodificação:")
    print("  - Valor 76001 → dummy_ethnic = 1")
    print("  - Outros valores → dummy_ethnic = 0")
    print("  - Valores negativos → missing")
    
    results = []
    
    for wave_dir, config in WAVES_CONFIG.items():
        result = process_wave(wave_dir, config)
        if result:
            results.append(result)
    
    # Resumo geral
    if results:
        print(f"\n{'='*70}")
        print("RESUMO GERAL")
        print(f"{'='*70}")
        print(f"\n{'Onda':<20} {'Ano':<8} {'Variável':<10} {'Total':<10} {'dummy=1':<12} {'dummy=0':<12} {'Missing':<10}")
        print("-" * 90)
        
        for r in results:
            print(f"{r['wave']:<20} {r['year']:<8} {r['variable']:<10} {r['total']:<10} "
                  f"{r['dummy_1']:<12} {r['dummy_0']:<12} {r['missing']:<10}")
        
        print("\n✓ Processamento concluído com sucesso!")
    else:
        print("\n⚠️  Nenhuma onda foi processada com sucesso.")

if __name__ == '__main__':
    main()



