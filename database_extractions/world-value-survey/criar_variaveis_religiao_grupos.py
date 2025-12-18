#!/usr/bin/env python3
"""
Script para criar variáveis GrupoEv e GrupoCat do World Value Survey - Brasil
Utiliza leitura direta de CSVs para evitar conflitos com indexes do banco.
"""

import csv
import os

# Configuração dos arquivos e variáveis
config = {
    'Wave2': {
        'file': 'wave-2-1991-br/WV2_Data_Brazil_Csv_v1.6.1.csv',
        'variable': 'V144',
        'year': 1991,
        'grupo_ev_code': 62,
        'grupo_cat_code': 64
    },
    'Wave3': {
        'file': 'wave-3-1997-br/WV3_Data_Brazil_Csv_v20221107.1.csv',
        'variable': 'V179G',
        'year': 1997,
        'grupo_ev_code': 2,
        'grupo_cat_code': 1
    },
    'Wave5': {
        'file': 'wave-5-2006-br/WV5_Data_Brazil_Csv_v20201117.1.csv',
        'variable': 'V185G',
        'year': 2005,
        'grupo_ev_code': 2,
        'grupo_cat_code': 1
    },
    'Wave6': {
        'file': 'wave-6-2014-br/WV6_Data_Brazil_Csv_v20221117.1.csv',
        'variable': 'V144',
        'year': 2014,
        'grupo_ev_code': 21400000,
        'grupo_cat_code': 10000000
    },
    'Wave7': {
        'file': 'wave-7-2018-br/WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'variable': 'Q289',
        'year': 2018,
        'grupo_ev_code': 2,
        'grupo_cat_code': 1
    }
}

# Diretório base
base_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print("CRIAÇÃO DE VARIÁVEIS GRUPOEV E GRUPOCAT - WORLD VALUE SURVEY - BRASIL")
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
    grupo_ev_code = info['grupo_ev_code']
    grupo_cat_code = info['grupo_cat_code']
    year = info['year']
    
    print(f"\n{'='*80}")
    print(f"{wave} - {year}: {os.path.basename(file_path)}")
    print(f"Variável: {variable}")
    print(f"Código GrupoEv: {grupo_ev_code}")
    print(f"Código GrupoCat: {grupo_cat_code}")
    print(f"{'='*80}")
    
    try:
        # Passo 1: Ler o CSV original usando csv.reader
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
                # Garantir que a linha tem o mesmo número de colunas do header
                if len(row) > len(header):
                    row = row[:len(header)]
                elif len(row) < len(header):
                    row.extend([''] * (len(header) - len(row)))
                rows.append(row)
        
        print(f"Total de linhas lidas: {len(rows)}")
        
        # Adicionar novas colunas ao header
        header.append('GrupoEv')
        header.append('GrupoCat')
        
        # Processar cada linha
        grupo_ev_index = len(header) - 2
        grupo_cat_index = len(header) - 1
        
        stats = {
            'total': len(rows),
            'negative_to_missing': 0,
            'grupo_ev_count': 0,
            'grupo_cat_count': 0,
            'other_values': 0,
            'missing': 0
        }
        
        # Determinar o tamanho esperado das linhas originais (sem as 2 novas colunas)
        original_header_size = len(header) - 2
        
        for row_idx, row in enumerate(rows):
            # Garantir que a linha tem o tamanho correto
            while len(row) < original_header_size:
                row.append('')
            if len(row) > original_header_size:
                row = row[:original_header_size]
            
            # Obter valor original
            original_value = row[var_index] if var_index < len(row) else ''
            
            # Passo 2: Converter valores negativos para missing
            num_value = None
            if is_negative(original_value):
                stats['negative_to_missing'] += 1
                grupo_ev_value = ''
                grupo_cat_value = ''
            else:
                # Converter para numérico
                num_value = convert_to_numeric(original_value)
                
                if num_value is None or num_value == '':
                    stats['missing'] += 1
                    grupo_ev_value = ''
                    grupo_cat_value = ''
                else:
                    # Passo 3: Criar variável GrupoEv
                    # Converter para int para garantir correspondência
                    int_value = int(num_value)
                    if int_value == grupo_ev_code:
                        grupo_ev_value = '1'
                        grupo_cat_value = '0'
                        stats['grupo_ev_count'] += 1
                    elif int_value == grupo_cat_code:
                        # Passo 4: Criar variável GrupoCat
                        grupo_ev_value = '0'
                        grupo_cat_value = '1'
                        stats['grupo_cat_count'] += 1
                    else:
                        # Outros valores
                        grupo_ev_value = '0'
                        grupo_cat_value = '0'
                        stats['other_values'] += 1
            
            # Adicionar novas variáveis à linha
            row.append(grupo_ev_value)
            row.append(grupo_cat_value)
        
        # Criar nome do arquivo de saída
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.dirname(file_path)
        output_file = os.path.join(output_dir, f"{base_name}_grupos_religiao.csv")
        
        # Escrever arquivo de saída
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)
            writer.writerows(rows)
        
        print(f"\nArquivo salvo: {output_file}")
        print(f"\nEstatísticas:")
        print(f"  Total de observações: {stats['total']}")
        print(f"  Valores negativos convertidos para missing: {stats['negative_to_missing']}")
        print(f"  GrupoEv = 1 (código {grupo_ev_code}): {stats['grupo_ev_count']} ({stats['grupo_ev_count']/stats['total']*100:.2f}%)")
        print(f"  GrupoCat = 1 (código {grupo_cat_code}): {stats['grupo_cat_count']} ({stats['grupo_cat_count']/stats['total']*100:.2f}%)")
        print(f"  Outros valores (ambos = 0): {stats['other_values']} ({stats['other_values']/stats['total']*100:.2f}%)")
        print(f"  Valores missing: {stats['missing']} ({stats['missing']/stats['total']*100:.2f}%)")
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {file_path}")
    except Exception as e:
        print(f"ERRO ao processar {wave}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("Processamento concluído!")
print("=" * 80)



