"""
Script para consolidar todos os dados processados do World Value Survey
em uma única base de dados final com mais de 7 mil registros.

Esta base final contém:
- id_ (sequencial de 1 até o último registro)
- id_usuario (V3 para waves 2,3,5,6 e D_interview para wave 7)
- ano_onda
- nome_onda
- Todas as variáveis processadas: y_moral_inv, aborto_z, homossex_z, prost_z,
  y_intitucional_4vars, congresso_z, justica_z, igreja_z, imprensa_z,
  freq_inv_z, pert_r_z, import_relig_z, relig_r, D_relig, D_N_relig,
  GrupoEv, GrupoCat
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent

# Configuração das ondas
WAVES_CONFIG = {
    'wave-2-1991-br': {
        'year': 1991,
        'nome_onda': 'Wave 2',
        'id_usuario_col': 'V3',
        'file_original': 'WV2_Data_Brazil_Csv_v1.6.1.csv',
        'file_y_moral_inv': 'WV2_Data_Brazil_Csv_v1.6.1_invertido.csv',
        'file_y_institucional': 'WV2_Data_Brazil_Csv_v1.6.1_institucional.csv',
        'file_freq_inv': 'WV2_Data_Brazil_Csv_v1.6.1_com_freq_inv.csv',
        'file_pert_r': 'WV2_Data_Brazil_Csv_v1.6.1_pert_r.csv',
        'file_import_relig': 'WV2_Data_Brazil_Csv_v1.6.1_com_import_relig_inv.csv',
        'file_relig_r': 'WV2_Data_Brazil_Csv_v1.6.1_relig_r.csv',
        'file_grupos': 'WV2_Data_Brazil_Csv_v1.6.1_grupos_religiao.csv',
        'file_dummy_ethnic': 'WV2_Data_Brazil_Csv_v1.6.1_com_dummy_ethnic.csv',
        'file_dummy_sex': 'WV2_Data_Brazil_Csv_v1.6.1_com_dummy_sex.csv',
        'file_escol_z': 'WV2_Data_Brazil_Csv_v1.6.1_com_escol_z.csv',
        'variables': {
            'aborto': 'V309',
            'homossex': 'V307',
            'prost': 'V308'
        }
    },
    'wave-3-1997-br': {
        'year': 1997,
        'nome_onda': 'Wave 3',
        'id_usuario_col': 'V3',
        'file_original': 'WV3_Data_Brazil_Csv_v20221107.1.csv',
        'file_y_moral_inv': 'WV3_Data_Brazil_Csv_v20221107.1_invertido.csv',
        'file_y_institucional': 'WV3_Data_Brazil_Csv_v20221107.1_institucional.csv',
        'file_freq_inv': 'WV3_Data_Brazil_Csv_v20221107.1_com_freq_inv.csv',
        'file_pert_r': 'WV3_Data_Brazil_Csv_v20221107.1_pert_r.csv',
        'file_import_relig': 'WV3_Data_Brazil_Csv_v20221107.1_com_import_relig_inv.csv',
        'file_relig_r': 'WV3_Data_Brazil_Csv_v20221107.1_relig_r.csv',
        'file_grupos': 'WV3_Data_Brazil_Csv_v20221107.1_grupos_religiao.csv',
        'file_dummy_ethnic': 'WV3_Data_Brazil_Csv_v20221107.1_com_dummy_ethnic.csv',
        'file_dummy_sex': 'WV3_Data_Brazil_Csv_v20221107.1_com_dummy_sex.csv',
        'file_escol_z': 'WV3_Data_Brazil_Csv_v20221107.1_com_escol_z.csv',
        'variables': {
            'aborto': 'V199',
            'homossex': 'V197',
            'prost': 'V198'
        }
    },
    'wave-5-2006-br': {
        'year': 2006,
        'nome_onda': 'Wave 5',
        'id_usuario_col': 'V3',
        'file_original': 'WV5_Data_Brazil_Csv_v20201117.1.csv',
        'file_y_moral_inv': 'WV5_Data_Brazil_Csv_v20201117.1_invertido.csv',
        'file_y_institucional': 'WV5_Data_Brazil_Csv_v20201117.1_institucional.csv',
        'file_freq_inv': 'WV5_Data_Brazil_Csv_v20201117.1_com_freq_inv.csv',
        'file_pert_r': 'WV5_Data_Brazil_Csv_v20201117.1_pert_r.csv',
        'file_import_relig': 'WV5_Data_Brazil_Csv_v20201117.1_com_import_relig_inv.csv',
        'file_relig_r': 'WV5_Data_Brazil_Csv_v20201117.1_relig_r.csv',
        'file_grupos': 'WV5_Data_Brazil_Csv_v20201117.1_grupos_religiao.csv',
        'file_dummy_ethnic': 'WV5_Data_Brazil_Csv_v20201117.1_com_dummy_ethnic.csv',
        'file_dummy_sex': 'WV5_Data_Brazil_Csv_v20201117.1_com_dummy_sex.csv',
        'file_escol_z': 'WV5_Data_Brazil_Csv_v20201117.1_com_escol_z.csv',
        'variables': {
            'aborto': 'V204',
            'homossex': 'V202',
            'prost': 'V203'
        }
    },
    'wave-6-2014-br': {
        'year': 2014,
        'nome_onda': 'Wave 6',
        'id_usuario_col': 'V3',
        'file_original': 'WV6_Data_Brazil_Csv_v20221117.1.csv',
        'file_y_moral_inv': 'WV6_Data_Brazil_Csv_v20221117.1_invertido.csv',
        'file_y_institucional': 'WV6_Data_Brazil_Csv_v20221117.1_institucional.csv',
        'file_freq_inv': 'WV6_Data_Brazil_Csv_v20221117.1_com_freq_inv.csv',
        'file_pert_r': 'WV6_Data_Brazil_Csv_v20221117.1_pert_r.csv',
        'file_import_relig': 'WV6_Data_Brazil_Csv_v20221117.1_com_import_relig_inv.csv',
        'file_relig_r': 'WV6_Data_Brazil_Csv_v20221117.1_relig_r.csv',
        'file_grupos': 'WV6_Data_Brazil_Csv_v20221117.1_grupos_religiao.csv',
        'file_dummy_ethnic': 'WV6_Data_Brazil_Csv_v20221117.1_com_dummy_ethnic.csv',
        'file_dummy_sex': 'WV6_Data_Brazil_Csv_v20221117.1_com_dummy_sex.csv',
        'file_escol_z': 'WV6_Data_Brazil_Csv_v20221117.1_com_escol_z.csv',
        'variables': {
            'aborto': 'V204',
            'homossex': 'V203',
            'prost': 'V203A'
        }
    },
    'wave-7-2018-br': {
        'year': 2018,
        'nome_onda': 'Wave 7',
        'id_usuario_col': 'D_INTERVIEW',
        'file_original': 'WVS_Wave_7_Brazil_Csv_v5.1.csv',
        'file_y_moral_inv': 'WVS_Wave_7_Brazil_Csv_v5.1_invertido.csv',
        'file_y_institucional': 'WVS_Wave_7_Brazil_Csv_v5.1_institucional.csv',
        'file_freq_inv': 'WVS_Wave_7_Brazil_Csv_v5.1_com_freq_inv.csv',
        'file_pert_r': 'WVS_Wave_7_Brazil_Csv_v5.1_pert_r.csv',
        'file_import_relig': 'WVS_Wave_7_Brazil_Csv_v5.1_com_import_relig_inv.csv',
        'file_relig_r': 'WVS_Wave_7_Brazil_Csv_v5.1_relig_r.csv',
        'file_grupos': 'WVS_Wave_7_Brazil_Csv_v5.1_grupos_religiao.csv',
        'file_dummy_ethnic': 'WVS_Wave_7_Brazil_Csv_v5.1_com_dummy_ethnic.csv',
        'file_dummy_sex': 'WVS_Wave_7_Brazil_Csv_v5.1_com_dummy_sex.csv',
        'file_escol_z': 'WVS_Wave_7_Brazil_Csv_v5.1_com_escol_z.csv',
        'variables': {
            'aborto': 'Q184',
            'homossex': 'Q182',
            'prost': 'Q183'
        }
    }
}

def load_csv_file(wave_dir, file_name):
    """Carrega um arquivo CSV, tentando diferentes separadores."""
    file_path = BASE_DIR / wave_dir / file_name
    
    if not file_path.exists():
        print(f"  ⚠️  Arquivo não encontrado: {file_path}")
        return None
    
    try:
        # Tenta primeiro com ponto-e-vírgula
        try:
            df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)
        except:
            # Se falhar, tenta com vírgula
            df = pd.read_csv(file_path, sep=',', encoding='utf-8', low_memory=False)
        
        # Remove aspas dos nomes das colunas se existirem
        df.columns = df.columns.str.strip('"').str.strip("'")
        
        return df
    except Exception as e:
        print(f"  ❌ Erro ao carregar {file_name}: {e}")
        return None

def process_wave(wave_name, config):
    """Processa uma onda completa e retorna DataFrame consolidado."""
    print(f"\n{'='*70}")
    print(f"PROCESSANDO: {wave_name.upper()}")
    print(f"{'='*70}")
    
    wave_dir = wave_name
    year = config['year']
    nome_onda = config['nome_onda']
    id_usuario_col = config['id_usuario_col']
    variables = config['variables']
    
    # Inicializar DataFrame final para esta onda
    df_final = pd.DataFrame()
    
    # 1. Carregar arquivo original para obter ID do usuário
    print(f"\n[1] Carregando arquivo original para ID do usuário...")
    df_original = load_csv_file(wave_dir, config['file_original'])
    if df_original is None:
        return None
    
    if id_usuario_col not in df_original.columns:
        print(f"  ❌ Coluna {id_usuario_col} não encontrada")
        return None
    
    df_final['id_usuario'] = df_original[id_usuario_col].copy()
    print(f"  ✓ ID do usuário carregado: {len(df_final)} registros")
    
    # 2. Carregar y_moral_inv e variáveis z-score
    print(f"\n[2] Carregando y_moral_inv e variáveis z-score...")
    df_y_moral = load_csv_file(wave_dir, config['file_y_moral_inv'])
    if df_y_moral is not None:
        # Extrair variáveis z-score
        aborto_var = f"{variables['aborto']}_inv_z"
        homossex_var = f"{variables['homossex']}_inv_z"
        prost_var = f"{variables['prost']}_inv_z"
        
        # Garantir que o DataFrame tem o mesmo número de linhas
        if len(df_y_moral) != len(df_final):
            print(f"  ⚠️  Número de linhas diferente: original={len(df_final)}, y_moral={len(df_y_moral)}")
            # Se os tamanhos são diferentes, vamos alinhar pelo índice
            df_y_moral = df_y_moral.reset_index(drop=True)
            df_final = df_final.reset_index(drop=True)
        
        if aborto_var in df_y_moral.columns:
            df_final['aborto_z'] = df_y_moral[aborto_var].values
        if homossex_var in df_y_moral.columns:
            df_final['homossex_z'] = df_y_moral[homossex_var].values
        if prost_var in df_y_moral.columns:
            df_final['prost_z'] = df_y_moral[prost_var].values
        if 'Y_moral_inv' in df_y_moral.columns:
            df_final['y_moral_inv'] = df_y_moral['Y_moral_inv'].values
        
        print(f"  ✓ Variáveis y_moral carregadas")
        print(f"    y_moral_inv: {df_final['y_moral_inv'].notna().sum()} valores válidos de {len(df_final)}")
    else:
        print(f"  ⚠️  Arquivo y_moral_inv não encontrado")
    
    # 3. Carregar y_institucional_4var
    # Como os arquivos individuais não foram salvos, vamos usar o arquivo institucional geral
    # e calcular as variáveis z-score das 4 variáveis comuns
    print(f"\n[3] Carregando y_institucional_4var...")
    df_y_inst = load_csv_file(wave_dir, config['file_y_institucional'])
    if df_y_inst is not None:
        # Mapear variáveis originais para cada onda (conforme VARS_COMUNS em recalcular_y_institucional_4variaveis.py)
        vars_map = {
            'wave-2-1991-br': {'congresso': 'V279', 'justica': 'V275', 'igreja': 'V272', 'imprensa': 'V276'},
            'wave-3-1997-br': {'congresso': 'V144', 'justica': 'V137', 'igreja': 'V135', 'imprensa': 'V138'},
            'wave-5-2006-br': {'congresso': 'V140', 'justica': 'V137', 'igreja': 'V131', 'imprensa': 'V133'},
            'wave-6-2014-br': {'congresso': 'V117', 'justica': 'V114', 'igreja': 'V108', 'imprensa': 'V110'},
            'wave-7-2018-br': {'congresso': 'Q73', 'justica': 'Q70', 'igreja': 'Q64', 'imprensa': 'Q66'}
        }
        
        if wave_name in vars_map:
            wave_vars = vars_map[wave_name]
            
            # Garantir alinhamento de índices
            df_y_inst = df_y_inst.reset_index(drop=True)
            df_final = df_final.reset_index(drop=True)
            
            # Procurar colunas z-score
            for var_name, var_col in wave_vars.items():
                z_col = f"{var_col}_z"
                if z_col in df_y_inst.columns:
                    df_final[f"{var_name}_z"] = df_y_inst[z_col].values
                else:
                    # Se não existe z-score, calcular
                    if var_col in df_y_inst.columns:
                        valid_values = df_y_inst[var_col].dropna()
                        if len(valid_values) > 0:
                            mean = valid_values.mean()
                            std = valid_values.std()
                            if std > 0:
                                df_final[f"{var_name}_z"] = ((df_y_inst[var_col] - mean) / std).values
            
            # Calcular y_institucional_4var se não existir
            z_vars = [f"{var_name}_z" for var_name in wave_vars.keys() if f"{var_name}_z" in df_final.columns]
            if len(z_vars) == 4:
                df_final['y_intitucional_4vars'] = df_final[z_vars].mean(axis=1, skipna=False)
            elif 'Y_institucional' in df_y_inst.columns:
                # Usar Y_institucional como fallback
                df_final['y_intitucional_4vars'] = df_y_inst['Y_institucional'].values
        
        print(f"  ✓ Variáveis y_institucional carregadas")
    else:
        print(f"  ⚠️  Arquivo y_institucional não encontrado")
    
    # 4. Carregar freq_inv_z
    print(f"\n[4] Carregando freq_inv_z...")
    df_freq = load_csv_file(wave_dir, config['file_freq_inv'])
    if df_freq is not None:
        # Garantir alinhamento de índices
        df_freq = df_freq.reset_index(drop=True)
        df_final = df_final.reset_index(drop=True)
        
        if 'Freq_inv_z' in df_freq.columns:
            df_final['freq_inv_z'] = df_freq['Freq_inv_z'].values
        elif 'freq_inv_z' in df_freq.columns:
            df_final['freq_inv_z'] = df_freq['freq_inv_z'].values
        
        valid_count = df_final['freq_inv_z'].notna().sum() if 'freq_inv_z' in df_final.columns else 0
        print(f"  ✓ freq_inv_z carregada ({valid_count} valores válidos de {len(df_final)})")
    else:
        print(f"  ⚠️  Arquivo freq_inv não encontrado")
    
    # 5. Carregar pert_r_z
    print(f"\n[5] Carregando pert_r_z...")
    df_pert = load_csv_file(wave_dir, config['file_pert_r'])
    if df_pert is not None:
        # Garantir alinhamento de índices
        df_pert = df_pert.reset_index(drop=True)
        df_final = df_final.reset_index(drop=True)
        
        if 'Pert_r_z' in df_pert.columns:
            df_final['pert_r_z'] = df_pert['Pert_r_z'].values
        elif 'pert_r_z' in df_pert.columns:
            df_final['pert_r_z'] = df_pert['pert_r_z'].values
        
        valid_count = df_final['pert_r_z'].notna().sum() if 'pert_r_z' in df_final.columns else 0
        print(f"  ✓ pert_r_z carregada ({valid_count} valores válidos de {len(df_final)})")
    else:
        print(f"  ⚠️  Arquivo pert_r não encontrado")
    
    # 6. Carregar import_relig_z
    print(f"\n[6] Carregando import_relig_z...")
    df_import = load_csv_file(wave_dir, config['file_import_relig'])
    if df_import is not None:
        # Garantir alinhamento de índices
        df_import = df_import.reset_index(drop=True)
        df_final = df_final.reset_index(drop=True)
        
        if 'import_relig_inv_z' in df_import.columns:
            df_final['import_relig_z'] = df_import['import_relig_inv_z'].values
        
        valid_count = df_final['import_relig_z'].notna().sum() if 'import_relig_z' in df_final.columns else 0
        print(f"  ✓ import_relig_z carregada ({valid_count} valores válidos de {len(df_final)})")
    else:
        print(f"  ⚠️  Arquivo import_relig não encontrado")
    
    # 7. Carregar relig_r, D_relig, D_N_relig
    print(f"\n[7] Carregando relig_r, D_relig, D_N_relig...")
    df_relig = load_csv_file(wave_dir, config['file_relig_r'])
    if df_relig is not None:
        # Garantir alinhamento de índices
        df_relig = df_relig.reset_index(drop=True)
        df_final = df_final.reset_index(drop=True)
        
        if 'relig_r' in df_relig.columns:
            df_final['relig_r'] = pd.to_numeric(df_relig['relig_r'], errors='coerce').values
        if 'D_relig' in df_relig.columns:
            df_final['D_relig'] = pd.to_numeric(df_relig['D_relig'], errors='coerce').values
        if 'D_N_relig' in df_relig.columns:
            df_final['D_N_relig'] = pd.to_numeric(df_relig['D_N_relig'], errors='coerce').values
        
        valid_count = df_final['relig_r'].notna().sum() if 'relig_r' in df_final.columns else 0
        print(f"  ✓ Variáveis relig_r carregadas ({valid_count} valores válidos de {len(df_final)})")
    else:
        print(f"  ⚠️  Arquivo relig_r não encontrado")
    
    # 8. Carregar GrupoEv e GrupoCat
    print(f"\n[8] Carregando GrupoEv e GrupoCat...")
    df_grupos = load_csv_file(wave_dir, config['file_grupos'])
    if df_grupos is not None:
        # Garantir alinhamento de índices
        df_grupos = df_grupos.reset_index(drop=True)
        df_final = df_final.reset_index(drop=True)
        
        if 'GrupoEv' in df_grupos.columns:
            df_final['GrupoEv'] = pd.to_numeric(df_grupos['GrupoEv'], errors='coerce').values
        if 'GrupoCat' in df_grupos.columns:
            df_final['GrupoCat'] = pd.to_numeric(df_grupos['GrupoCat'], errors='coerce').values
        
        valid_count = df_final['GrupoEv'].notna().sum() if 'GrupoEv' in df_final.columns else 0
        print(f"  ✓ Variáveis grupos religiosos carregadas ({valid_count} valores válidos de {len(df_final)})")
    else:
        print(f"  ⚠️  Arquivo grupos não encontrado")
    
    # 9. Carregar dummy_ethnic
    print(f"\n[9] Carregando dummy_ethnic...")
    df_dummy_eth = load_csv_file(wave_dir, config['file_dummy_ethnic'])
    if df_dummy_eth is not None:
        # Garantir alinhamento de índices
        df_dummy_eth = df_dummy_eth.reset_index(drop=True)
        df_final = df_final.reset_index(drop=True)
        
        if 'dummy_ethnic' in df_dummy_eth.columns:
            df_final['dummy_ethnic'] = pd.to_numeric(df_dummy_eth['dummy_ethnic'], errors='coerce').values
        
        valid_count = df_final['dummy_ethnic'].notna().sum() if 'dummy_ethnic' in df_final.columns else 0
        print(f"  ✓ dummy_ethnic carregada ({valid_count} valores válidos de {len(df_final)})")
    else:
        print(f"  ⚠️  Arquivo dummy_ethnic não encontrado")
    
    # 10. Carregar dummy_sex
    print(f"\n[10] Carregando dummy_sex...")
    df_dummy_sex = load_csv_file(wave_dir, config['file_dummy_sex'])
    if df_dummy_sex is not None:
        # Garantir alinhamento de índices
        df_dummy_sex = df_dummy_sex.reset_index(drop=True)
        df_final = df_final.reset_index(drop=True)
        
        if 'dummy_sex' in df_dummy_sex.columns:
            df_final['dummy_sex'] = pd.to_numeric(df_dummy_sex['dummy_sex'], errors='coerce').values
        
        valid_count = df_final['dummy_sex'].notna().sum() if 'dummy_sex' in df_final.columns else 0
        print(f"  ✓ dummy_sex carregada ({valid_count} valores válidos de {len(df_final)})")
    else:
        print(f"  ⚠️  Arquivo dummy_sex não encontrado")
    
    # 11. Carregar escol_z
    print(f"\n[11] Carregando escol_z...")
    df_escol = load_csv_file(wave_dir, config['file_escol_z'])
    if df_escol is not None:
        # Garantir alinhamento de índices
        df_escol = df_escol.reset_index(drop=True)
        df_final = df_final.reset_index(drop=True)
        
        if 'escol_z' in df_escol.columns:
            df_final['escol_z'] = df_escol['escol_z'].values
        
        valid_count = df_final['escol_z'].notna().sum() if 'escol_z' in df_final.columns else 0
        print(f"  ✓ escol_z carregada ({valid_count} valores válidos de {len(df_final)})")
    else:
        print(f"  ⚠️  Arquivo escol_z não encontrado")
    
    # Adicionar informações da onda
    df_final['ano_onda'] = year
    df_final['nome_onda'] = nome_onda
    
    print(f"\n✓ Onda {wave_name} processada: {len(df_final)} registros")
    print(f"  Colunas: {', '.join(df_final.columns)}")
    
    return df_final

def main():
    """Função principal."""
    print("="*70)
    print("CONSOLIDAÇÃO DA BASE DE DADOS FINAL - WORLD VALUE SURVEY")
    print("="*70)
    
    all_dataframes = []
    
    # Processar cada onda
    for wave_name, config in WAVES_CONFIG.items():
        df_wave = process_wave(wave_name, config)
        if df_wave is not None:
            all_dataframes.append(df_wave)
    
    if not all_dataframes:
        print("\n❌ Nenhum dado foi processado!")
        return
    
    # Empilhar todos os DataFrames
    print(f"\n{'='*70}")
    print("EMPILHANDO TODOS OS DADOS")
    print("="*70)
    
    # Garantir que todas as colunas existam em todos os DataFrames
    all_columns = set()
    for df in all_dataframes:
        all_columns.update(df.columns)
    
    # Adicionar colunas faltantes como NaN
    for df in all_dataframes:
        for col in all_columns:
            if col not in df.columns:
                df[col] = np.nan
    
    # Ordenar colunas de forma consistente
    column_order = [
        'id_', 'id_usuario', 'ano_onda', 'nome_onda',
        'y_moral_inv', 'aborto_z', 'homossex_z', 'prost_z',
        'y_intitucional_4vars', 'congresso_z', 'justica_z', 'igreja_z', 'imprensa_z',
        'freq_inv_z', 'pert_r_z', 'import_relig_z',
        'relig_r', 'D_relig', 'D_N_relig', 'GrupoEv', 'GrupoCat',
        'dummy_ethnic', 'dummy_sex', 'escol_z'
    ]
    
    # Adicionar outras colunas que possam existir
    for col in all_columns:
        if col not in column_order:
            column_order.append(col)
    
    # Reordenar colunas em cada DataFrame
    for i, df in enumerate(all_dataframes):
        # Manter apenas colunas que existem no DataFrame
        existing_cols = [col for col in column_order if col in df.columns]
        all_dataframes[i] = df[existing_cols]
    
    # Concatenar todos os DataFrames
    df_final = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    # Criar ID sequencial
    df_final.insert(0, 'id_', range(1, len(df_final) + 1))
    
    # Reordenar colunas finais
    final_columns = ['id_', 'id_usuario', 'ano_onda', 'nome_onda']
    for col in column_order:
        if col in df_final.columns and col not in final_columns:
            final_columns.append(col)
    
    # Adicionar outras colunas que possam ter sido criadas
    for col in df_final.columns:
        if col not in final_columns:
            final_columns.append(col)
    
    df_final = df_final[final_columns]
    
    print(f"\n✓ Total de registros consolidados: {len(df_final)}")
    print(f"✓ Total de colunas: {len(df_final.columns)}")
    print(f"\nColunas principais:")
    for col in final_columns[:20]:
        print(f"  - {col}")
    if len(final_columns) > 20:
        print(f"  ... e mais {len(final_columns) - 20} colunas")
    
    # Estatísticas descritivas
    print(f"\n{'─'*70}")
    print("ESTATÍSTICAS POR ONDA:")
    print(f"{'─'*70}")
    for nome in df_final['nome_onda'].unique():
        df_onda = df_final[df_final['nome_onda'] == nome]
        print(f"\n{nome}:")
        print(f"  Registros: {len(df_onda)}")
        print(f"  Ano: {df_onda['ano_onda'].iloc[0]}")
    
    # Salvar arquivo final
    output_file = BASE_DIR / 'base_dados_final_consolidada.csv'
    df_final.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    print(f"\n{'='*70}")
    print("BASE DE DADOS FINAL SALVA!")
    print("="*70)
    print(f"\nArquivo: {output_file}")
    print(f"Total de registros: {len(df_final)}")
    print(f"Total de colunas: {len(df_final.columns)}")
    
    # Verificar valores missing
    print(f"\n{'─'*70}")
    print("VALORES MISSING POR VARIÁVEL:")
    print(f"{'─'*70}")
    main_vars = ['y_moral_inv', 'aborto_z', 'homossex_z', 'prost_z',
                 'y_intitucional_4vars', 'congresso_z', 'justica_z', 'igreja_z', 'imprensa_z',
                 'freq_inv_z', 'pert_r_z', 'import_relig_z',
                 'relig_r', 'D_relig', 'D_N_relig', 'GrupoEv', 'GrupoCat',
                 'dummy_ethnic', 'dummy_sex', 'escol_z']
    
    for var in main_vars:
        if var in df_final.columns:
            missing = df_final[var].isna().sum()
            pct = (missing / len(df_final)) * 100
            print(f"{var:25s}: {missing:5d} ({pct:5.2f}%)")

if __name__ == "__main__":
    main()

