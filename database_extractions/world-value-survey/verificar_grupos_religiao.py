#!/usr/bin/env python3
"""Script para verificar se as variáveis GrupoEv e GrupoCat foram criadas corretamente"""

import csv
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Verificar Wave 7
file_path = os.path.join(base_dir, 'wave-7-2018-br/WVS_Wave_7_Brazil_Csv_v5.1_grupos_religiao.csv')

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=';')
    header = next(reader)
    q289_idx = header.index('Q289')
    grupoev_idx = header.index('GrupoEv')
    grupocat_idx = header.index('GrupoCat')
    
    rows = list(reader)
    
    print("Wave 7 - Verificação:")
    print(f"Total de linhas: {len(rows)}")
    
    # Casos onde Q289 = 1 (deve ser GrupoCat = 1)
    casos_q289_1 = [row for row in rows if row[q289_idx] in ['1', '1.0']]
    print(f"\nCasos onde Q289 = 1 (GrupoCat esperado): {len(casos_q289_1)}")
    if casos_q289_1:
        print(f"  Primeiro caso: Q289={casos_q289_1[0][q289_idx]}, GrupoEv={casos_q289_1[0][grupoev_idx]}, GrupoCat={casos_q289_1[0][grupocat_idx]}")
        # Verificar se todos têm GrupoCat = 1
        todos_cat = all(row[grupocat_idx] == '1' for row in casos_q289_1)
        print(f"  Todos têm GrupoCat = 1? {todos_cat}")
    
    # Casos onde Q289 = 2 (deve ser GrupoEv = 1)
    casos_q289_2 = [row for row in rows if row[q289_idx] in ['2', '2.0']]
    print(f"\nCasos onde Q289 = 2 (GrupoEv esperado): {len(casos_q289_2)}")
    if casos_q289_2:
        print(f"  Primeiro caso: Q289={casos_q289_2[0][q289_idx]}, GrupoEv={casos_q289_2[0][grupoev_idx]}, GrupoCat={casos_q289_2[0][grupocat_idx]}")
        # Verificar se todos têm GrupoEv = 1
        todos_ev = all(row[grupoev_idx] == '1' for row in casos_q289_2)
        print(f"  Todos têm GrupoEv = 1? {todos_ev}")

# Verificar Wave 2
file_path = os.path.join(base_dir, 'wave-2-1991-br/WV2_Data_Brazil_Csv_v1.6.1_grupos_religiao.csv')

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=';')
    header = next(reader)
    v144_idx = header.index('V144')
    grupoev_idx = header.index('GrupoEv')
    grupocat_idx = header.index('GrupoCat')
    
    rows = list(reader)
    
    print("\n\nWave 2 - Verificação:")
    print(f"Total de linhas: {len(rows)}")
    
    # Casos onde V144 = 62 (deve ser GrupoEv = 1)
    casos_v144_62 = [row for row in rows if row[v144_idx] in ['62', '62.0']]
    print(f"\nCasos onde V144 = 62 (GrupoEv esperado): {len(casos_v144_62)}")
    if casos_v144_62:
        print(f"  Primeiro caso: V144={casos_v144_62[0][v144_idx]}, GrupoEv={casos_v144_62[0][grupoev_idx]}, GrupoCat={casos_v144_62[0][grupocat_idx]}")
        todos_ev = all(row[grupoev_idx] == '1' for row in casos_v144_62)
        print(f"  Todos têm GrupoEv = 1? {todos_ev}")
    
    # Casos onde V144 = 64 (deve ser GrupoCat = 1)
    casos_v144_64 = [row for row in rows if row[v144_idx] in ['64', '64.0']]
    print(f"\nCasos onde V144 = 64 (GrupoCat esperado): {len(casos_v144_64)}")
    if casos_v144_64:
        print(f"  Primeiro caso: V144={casos_v144_64[0][v144_idx]}, GrupoEv={casos_v144_64[0][grupoev_idx]}, GrupoCat={casos_v144_64[0][grupocat_idx]}")
        todos_cat = all(row[grupocat_idx] == '1' for row in casos_v144_64)
        print(f"  Todos têm GrupoCat = 1? {todos_cat}")

print("\nVerificação concluída!")



