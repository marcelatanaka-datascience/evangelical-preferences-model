#!/usr/bin/env python3
"""
Script para calcular frequências simples de variáveis específicas
nos arquivos CSV do World Value Survey - Brasil
Usa csv reader nativo para evitar problemas com pandas
"""

import csv
import os
from collections import Counter

# Configuração dos arquivos e variáveis
config = {
    'Wave2': {
        'file': 'wave-2-1991-br/WV2_Data_Brazil_Csv_v1.6.1.csv',
        'variable': 'V151'
    },
    'Wave3': {
        'file': 'wave-3-1997-br/WV3_Data_Brazil_Csv_v20221107.1.csv',
        'variable': 'V182'
    },
    'Wave5': {
        'file': 'wave-5-2006-br/WV5_Data_Brazil_Csv_v20201117.1.csv',
        'variable': 'V187'
    },
    'Wave6': {
        'file': 'wave-6-2014-br/WV6_Data_Brazil_Csv_v20221117.1.csv',
        'variable': 'V147'
    },
    'Wave7': {
        'file': 'wave-7-2018-br/WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'variable': 'Q173'
    }
}

# Diretório base
base_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 80)
print("FREQUÊNCIAS SIMPLES - WORLD VALUE SURVEY - BRASIL")
print("=" * 80)
print()

for wave, info in config.items():
    file_path = os.path.join(base_dir, info['file'])
    variable = info['variable']
    
    print(f"\n{'='*80}")
    print(f"{wave}: {os.path.basename(file_path)}")
    print(f"Variável: {variable}")
    print(f"{'='*80}")
    
    try:
        # Ler o CSV diretamente com csv.reader
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            
            # Ler o cabeçalho
            header = next(reader)
            
            # Encontrar o índice da variável
            try:
                var_index = header.index(variable)
            except ValueError:
                print(f"ERRO: Variável {variable} não encontrada no arquivo!")
                print(f"Variáveis disponíveis (primeiras 10): {header[:10]}")
                continue
            
            # Ler todos os valores da variável
            values = []
            total_rows = 0
            for row in reader:
                total_rows += 1
                if len(row) > var_index:
                    value = row[var_index].strip()
                    values.append(value)
                else:
                    values.append('')
            
            # Calcular frequências
            freq_counter = Counter(values)
            
            # Separar valores válidos e missing
            valid_values = {k: v for k, v in freq_counter.items() if k != ''}
            missing_count = freq_counter.get('', 0)
            
            # Calcular total de valores válidos
            total_valid = sum(valid_values.values())
            
            # Ordenar valores (tentando converter para numérico quando possível)
            def sort_key(x):
                try:
                    return float(x)
                except ValueError:
                    return x
            
            sorted_values = sorted(valid_values.keys(), key=sort_key)
            
            # Exibir resultados
            print(f"\nTotal de observações: {total_rows}")
            print(f"Observações válidas: {total_valid}")
            if missing_count > 0:
                print(f"Observações missing/vazias: {missing_count}")
            print()
            
            # Criar tabela de resultados
            print(f"{'Valor':<15} {'Frequência':<15} {'Percentual':<15}")
            print("-" * 45)
            
            for value in sorted_values:
                count = valid_values[value]
                percent = (count / total_valid * 100) if total_valid > 0 else 0
                print(f"{value:<15} {count:<15} {percent:<15.2f}")
            
            # Adicionar linha de total
            print("-" * 45)
            print(f"{'Total':<15} {total_valid:<15} {'100.00':<15}")
            
            if missing_count > 0:
                print(f"{'Missing/NA':<15} {missing_count:<15} {round(missing_count / total_rows * 100, 2):<15.2f}")
                print(f"{'Total (com missing)':<15} {total_rows:<15} {'100.00':<15}")
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado: {file_path}")
    except Exception as e:
        print(f"ERRO ao processar {wave}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("Análise concluída!")
print("=" * 80)
