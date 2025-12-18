#!/usr/bin/env python3
"""
Script para recodificar variáveis de religião do World Value Survey - Brasil
Cria variáveis relig_r, D_relig e D_N_relig para cada onda
"""

import csv
import os

# Configuração dos arquivos e variáveis
config = {
    'Wave2': {
        'file': 'wave-2-1991-br/WV2_Data_Brazil_Csv_v1.6.1.csv',
        'variable': 'V151',
        'recoding': {
            1: 1,  # Original 1 -> relig_r: 1
            2: 0,  # Original 2 -> relig_r: 0
            3: 0   # Original 3 -> relig_r: 0
        }
    },
    'Wave3': {
        'file': 'wave-3-1997-br/WV3_Data_Brazil_Csv_v20221107.1.csv',
        'variable': 'V182',
        'recoding': {
            1: 1,  # Original 1 -> relig_r: 1
            2: 0,  # Original 2 -> relig_r: 0
            3: 0,  # Original 3 -> relig_r: 0
            4: 0   # Original 4 -> relig_r: 0
        }
    },
    'Wave5': {
        'file': 'wave-5-2006-br/WV5_Data_Brazil_Csv_v20201117.1.csv',
        'variable': 'V187',
        'recoding': {
            1: 1,  # Original 1 -> relig_r: 1
            2: 0,  # Original 2 -> relig_r: 0
            3: 0,  # Original 3 -> relig_r: 0
            4: 0   # Original 4 -> relig_r: 0
        }
    },
    'Wave6': {
        'file': 'wave-6-2014-br/WV6_Data_Brazil_Csv_v20221117.1.csv',
        'variable': 'V147',
        'recoding': {
            1: 1,  # Original 1 -> relig_r: 1
            2: 0,  # Original 2 -> relig_r: 0
            3: 0,  # Original 3 -> relig_r: 0
            4: 0   # Original 4 -> relig_r: 0
        }
    },
    'Wave7': {
        'file': 'wave-7-2018-br/WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'variable': 'Q173',
        'recoding': {
            1: 1,  # Original 1 -> relig_r: 1
            2: 0,  # Original 2 -> relig_r: 0
            3: 0,  # Original 3 -> relig_r: 0
            4: 0   # Original 4 -> relig_r: 0
        }
    }
}

# Diretório base
base_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print("RECODIFICAÇÃO DE VARIÁVEIS DE RELIGIÃO - WORLD VALUE SURVEY - BRASIL")
print("=" * 80)
print()

def convert_to_numeric(value):
    """Converte valor para numérico, retorna None se não for possível"""
    if value == '' or value is None:
        return None
    try:
        # Remove aspas se houver
        value = value.strip().strip('"').strip("'")
        num_value = float(value)
        return int(num_value) if num_value.is_integer() else num_value
    except (ValueError, AttributeError):
        return None

def is_negative(value):
    """Verifica se o valor é negativo"""
    num_value = convert_to_numeric(value)
    if num_value is None:
        return False
    return num_value < 0

for wave, info in config.items():
    file_path = os.path.join(base_dir, info['file'])
    variable = info['variable']
    recoding = info['recoding']
    
    print(f"\n{'='*80}")
    print(f"{wave}: {os.path.basename(file_path)}")
    print(f"Variável: {variable}")
    print(f"{'='*80}")
    
    try:
        # Ler o CSV original
        rows = []
        header = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader)
            
            # Encontrar o índice da variável
            try:
                var_index = header.index(variable)
            except ValueError:
                print(f"ERRO: Variável {variable} não encontrada no arquivo!")
                print(f"Variáveis disponíveis (primeiras 20): {header[:20]}")
                continue
            
            print(f"Variável {variable} encontrada no índice {var_index}")
            
            # Ler todas as linhas
            for row in reader:
                rows.append(row)
        
        print(f"Total de linhas lidas: {len(rows)}")
        
        # Adicionar novas colunas ao header
        header.append('relig_r')
        header.append('D_relig')
        header.append('D_N_relig')
        
        # Processar cada linha
        relig_r_index = len(header) - 3
        d_relig_index = len(header) - 2
        d_n_relig_index = len(header) - 1
        
        stats = {
            'total': len(rows),
            'negative_to_missing': 0,
            'recoded': 0,
            'missing': 0,
            'recoding_counts': {}
        }
        
        # Determinar o tamanho esperado das linhas originais (sem as 3 novas colunas)
        original_header_size = len(header) - 3
        
        for row_idx, row in enumerate(rows):
            # Remover colunas extras vazias no final (se houver)
            while len(row) > original_header_size and row[-1] == '':
                row.pop()
            
            # Garantir que a linha tem o tamanho correto (preencher com vazios se necessário)
            while len(row) < original_header_size:
                row.append('')
            
            # Obter valor original
            original_value = row[var_index] if var_index < len(row) else ''
            
            # Passo 2: Converter valores negativos para missing
            num_value = None
            if is_negative(original_value):
                stats['negative_to_missing'] += 1
                relig_r_value = ''
            else:
                # Converter para numérico
                num_value = convert_to_numeric(original_value)
                
                if num_value is None or num_value == '':
                    stats['missing'] += 1
                    relig_r_value = ''
                else:
                    # Passo 3: Recodificar
                    # Converter para int para garantir correspondência com as chaves do dicionário
                    int_value = int(num_value)
                    if int_value in recoding:
                        relig_r_value = str(recoding[int_value])
                        stats['recoded'] += 1
                        stats['recoding_counts'][relig_r_value] = stats['recoding_counts'].get(relig_r_value, 0) + 1
                    else:
                        # Valor não mapeado, manter como missing
                        stats['missing'] += 1
                        relig_r_value = ''
            
            # Adicionar relig_r à linha
            row.append(relig_r_value)
            
            # Passo 4: Criar variáveis dummy
            if relig_r_value == '1':
                d_relig = '1'
                d_n_relig = '0'
            elif relig_r_value == '0':
                d_relig = '0'
                d_n_relig = '1'
            else:
                # Missing - ambas ficam como missing
                d_relig = ''
                d_n_relig = ''
            
            row.append(d_relig)
            row.append(d_n_relig)
        
        # Criar nome do arquivo de saída
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.dirname(file_path)
        output_file = os.path.join(output_dir, f"{base_name}_relig_r.csv")
        
        # Escrever arquivo de saída
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            writer.writerows(rows)
        
        print(f"\nArquivo salvo: {output_file}")
        print(f"\nEstatísticas:")
        print(f"  Total de observações: {stats['total']}")
        print(f"  Valores negativos convertidos para missing: {stats['negative_to_missing']}")
        print(f"  Valores recodificados: {stats['recoded']}")
        print(f"  Valores missing: {stats['missing']}")
        print(f"\nDistribuição de relig_r:")
        for value, count in sorted(stats['recoding_counts'].items()):
            print(f"  relig_r = {value}: {count} ({count/stats['total']*100:.2f}%)")
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {file_path}")
    except Exception as e:
        print(f"ERRO ao processar {wave}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("Processamento concluído!")
print("=" * 80)

