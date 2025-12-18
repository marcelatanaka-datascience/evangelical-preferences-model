"""
Script para calcular a frequência da variável Q290 da wave 7 do World Values Survey.
Usa leitura manual de CSV para evitar problemas com o índice do banco.
"""

import csv
from collections import Counter
from pathlib import Path

# Caminho do arquivo
BASE_DIR = Path(__file__).parent
CSV_FILE = BASE_DIR / 'wave-7-2018-br' / 'WVS_Wave_7_Brazil_Csv_v5.1.csv'

# Encontrar o índice da coluna Q290
def find_q290_index(header_row):
    """Encontra o índice da coluna Q290 no cabeçalho."""
    for i, col in enumerate(header_row):
        # Remove aspas e espaços
        col_clean = col.strip('"').strip()
        if col_clean == 'Q290':
            return i
    return None

# Ler o CSV manualmente
print(f"Lendo arquivo: {CSV_FILE}")
print("-" * 60)

q290_index = None
q290_values = []

with open(CSV_FILE, 'r', encoding='utf-8') as f:
    # Usar csv.reader com delimitador ponto-e-vírgula
    reader = csv.reader(f, delimiter=';', quotechar='"')
    
    # Ler o cabeçalho
    header = next(reader)
    q290_index = find_q290_index(header)
    
    if q290_index is None:
        print("ERRO: Coluna Q290 não encontrada no arquivo!")
        print("Colunas disponíveis:", header[:20], "...")
        exit(1)
    
    print(f"Coluna Q290 encontrada no índice: {q290_index}")
    print(f"Total de colunas: {len(header)}")
    print("-" * 60)
    
    # Ler todas as linhas de dados
    for row_num, row in enumerate(reader, start=2):  # start=2 porque já lemos o header
        if len(row) > q290_index:
            value = row[q290_index].strip('"').strip()
            # Converter para float se possível, senão manter como string
            try:
                if value:
                    q290_values.append(float(value))
                else:
                    q290_values.append(None)
            except ValueError:
                q290_values.append(value)
        else:
            print(f"Aviso: Linha {row_num} não tem coluna Q290 (apenas {len(row)} colunas)")

print(f"Total de registros lidos: {len(q290_values)}")
print("-" * 60)

# Calcular frequências
print("\nFREQUÊNCIA DA VARIÁVEL Q290 (Wave 7 - 2018)")
print("=" * 60)

# Contar frequências
frequencies = Counter(q290_values)

# Ordenar por valor (se numérico) ou alfabeticamente
try:
    # Tentar ordenar numericamente
    sorted_items = sorted(frequencies.items(), key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))
except TypeError:
    # Se não for possível ordenar numericamente, ordenar alfabeticamente
    sorted_items = sorted(frequencies.items(), key=lambda x: (x[0] is None, str(x[0])))

# Calcular totais
total_valid = sum(count for value, count in frequencies.items() if value is not None)
total_missing = frequencies.get(None, 0)
total_all = len(q290_values)

# Imprimir tabela de frequências
print(f"\n{'Valor':<15} {'Frequência':<15} {'Percentual':<15} {'Percentual Válido':<20}")
print("-" * 65)

for value, count in sorted_items:
    if value is None:
        label = "Missing/NA"
        pct = (count / total_all * 100) if total_all > 0 else 0
        pct_valid = "-"
    else:
        label = str(value)
        pct = (count / total_all * 100) if total_all > 0 else 0
        pct_valid = (count / total_valid * 100) if total_valid > 0 else 0
    
    print(f"{label:<15} {count:<15} {pct:>6.2f}%{'':<8} {pct_valid:>6.2f}%{'':<13}" if value is not None else f"{label:<15} {count:<15} {pct:>6.2f}%{'':<8} {pct_valid:<20}")

print("-" * 65)
print(f"{'TOTAL':<15} {total_all:<15} {100.0:>6.2f}%{'':<8} {'':<20}")
print(f"{'Válidos':<15} {total_valid:<15} {(total_valid/total_all*100) if total_all > 0 else 0:>6.2f}%{'':<8} {100.0:>6.2f}%{'':<13}")
print(f"{'Missing':<15} {total_missing:<15} {(total_missing/total_all*100) if total_all > 0 else 0:>6.2f}%{'':<8} {'':<20}")

# Estatísticas descritivas (apenas para valores numéricos válidos)
numeric_values = [v for v in q290_values if v is not None and isinstance(v, (int, float)) and v >= 0]
if numeric_values:
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS DESCRITIVAS (valores numéricos válidos)")
    print("=" * 60)
    print(f"N: {len(numeric_values)}")
    print(f"Média: {sum(numeric_values) / len(numeric_values):.2f}")
    print(f"Mediana: {sorted(numeric_values)[len(numeric_values)//2]:.2f}")
    print(f"Mínimo: {min(numeric_values)}")
    print(f"Máximo: {max(numeric_values)}")

print("\n" + "=" * 60)
print("Análise concluída!")



