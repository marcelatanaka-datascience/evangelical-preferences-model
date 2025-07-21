import os
import time
import pandas as pd
import requests
import json
from PyPDF2 import PdfReader
import logging
from datetime import datetime
import signal
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ========= CONFIGURAÇÕES =========
# Caminhos
pdf_dir = "./questionarios"
csv_path = "pesquisas_cesop.csv"
output_csv_path = "pesquisas_cesop_com_religiao_opt.csv"
progress_file = "progresso_analise.json"
log_file = "analise_pdfs.log"

# Configurações do Llama/Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

# Configurações de processamento OTIMIZADAS
BATCH_SIZE = 30  # Aumentado drasticamente
TIMEOUT_SECONDS = 15  # Reduzido - IA deve responder rapidamente
MAX_RETRIES = 2  # Reduzido - menos tentativas
TEXTO_LIMITE = 20000  # Reduzido - texto menor = mais rápido
WORKERS = 4  # Processamento paralelo

# ========= CONFIGURAÇÃO DE LOGGING =========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========= FRASES DE BUSCA ULTRA RIGOROSAS =========
# Padrões muito específicos para evitar falsos positivos
frases_chave_compiladas = [
    # Perguntas diretas exatas
    re.compile(r"qual\s+(?:é\s+)?(?:a\s+)?sua\s+religião\s*[?:]", re.IGNORECASE),
    re.compile(r"você\s+tem\s+alguma\s+religião\s*[?:]", re.IGNORECASE),
    re.compile(r"qual\s+(?:a\s+)?sua\s+denominação\s+religiosa\s*[?:]", re.IGNORECASE),
    re.compile(r"frequenta\s+(?:algum\s+)?(?:local|lugar)\s+religioso\s*[?:]", re.IGNORECASE),
    re.compile(r"frequenta\s+(?:cultos?|missas?)\s*[?:]", re.IGNORECASE),
    re.compile(r"com\s+que\s+frequência.*(?:igreja|missa|culto|templo)", re.IGNORECASE),
    
    # Variáveis de questionário muito específicas
    re.compile(r"religião\s+do\s+entrevistado\s*[:.]", re.IGNORECASE),
    re.compile(r"denominação\s+religiosa\s+do\s+entrevistado", re.IGNORECASE),
    
    # Opções de resposta muito claras
    re.compile(r"católic[oa]\s*\(\s*\)\s*evangélic[oa]\s*\(\s*\)", re.IGNORECASE),
    re.compile(r"sem\s+religião\s*\(\s*\)", re.IGNORECASE),
    re.compile(r"você\s+se\s+considera\s*[:.]?\s*\(\s*\)\s*católic", re.IGNORECASE),
    
    # Escalas de frequência religiosa
    re.compile(r"nunca\s*\(\s*\)\s*raramente.*igreja", re.IGNORECASE),
    re.compile(r"sempre\s*\(\s*\).*às\s+vezes.*missa", re.IGNORECASE),
]

# Cache de textos já processados
texto_cache = {}
cache_lock = threading.Lock()

# ========= VARIÁVEIS GLOBAIS =========
progresso = {
    "ultimo_processado": -1,
    "total_com_religiao": 0,
    "total_processados": 0,
    "inicio_processamento": None,
    "batches_completados": 0,
    "falsos_positivos_evitados": 0
}

# ========= FUNÇÕES DE CONTROLE =========
def salvar_progresso():
    """Salva o progresso atual em arquivo JSON"""
    with open(progress_file, 'w') as f:
        json.dump(progresso, f, indent=2)

def carregar_progresso():
    """Carrega o progresso salvo"""
    global progresso
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progresso.update(json.load(f))
            logger.info(f"Progresso carregado: {progresso['total_processados']} PDFs já processados")
        except Exception as e:
            logger.warning(f"Erro ao carregar progresso: {e}")

def signal_handler(signum, frame):
    """Manipula interrupção do usuário (Ctrl+C)"""
    logger.info("Interrupção detectada. Salvando progresso...")
    salvar_progresso()
    sys.exit(0)

# ========= FUNÇÕES DE RESET =========
def reset_progresso():
    """Reseta todo o progresso e arquivos de resultado"""
    arquivos_para_deletar = [progress_file, output_csv_path, log_file]
    
    for arquivo in arquivos_para_deletar:
        if os.path.exists(arquivo):
            os.remove(arquivo)
            logger.info(f"🗑️  Arquivo removido: {arquivo}")
    
    global progresso
    progresso = {
        "ultimo_processado": -1,
        "total_com_religiao": 0,
        "total_processados": 0,
        "inicio_processamento": None,
        "batches_completados": 0,
        "falsos_positivos_evitados": 0
    }
    
    logger.info("✅ Progresso resetado com sucesso!")

def verificar_ollama():
    """Verifica se o Ollama está rodando e se o modelo está disponível"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            if MODEL_NAME in model_names or f"{MODEL_NAME}:latest" in model_names:
                logger.info(f"✅ Ollama e modelo {MODEL_NAME} disponíveis")
                return True
            else:
                logger.error(f"❌ Modelo {MODEL_NAME} não encontrado. Disponíveis: {model_names}")
                return False
        else:
            logger.error("❌ Ollama não está respondendo")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erro ao conectar com Ollama: {e}")
        return False

def ler_pdf_texto_cached(pdf_path):
    """Extrai texto do PDF com cache para evitar reler"""
    with cache_lock:
        if pdf_path in texto_cache:
            return texto_cache[pdf_path]
    
    try:
        reader = PdfReader(pdf_path)
        texto = ""
        # Pega apenas as primeiras 5 páginas para velocidade
        for page_num, page in enumerate(reader.pages[:5]):
            try:
                texto += page.extract_text() or ""
            except Exception:
                continue
        
        texto_lower = texto.lower()
        
        with cache_lock:
            texto_cache[pdf_path] = texto_lower
            # Limita cache a 1000 entradas
            if len(texto_cache) > 1000:
                oldest_key = next(iter(texto_cache))
                del texto_cache[oldest_key]
        
        return texto_lower
    except Exception as e:
        logger.warning(f"Erro lendo PDF {pdf_path}: {e}")
        return ""

def buscar_palavras_chave_otimizada(texto):
    """Busca ultra rigorosa usando regex muito específicas"""
    # Primeiro, exclui contextos que não são perguntas
    contextos_excluir = [
        r"liberdade\s+(?:de\s+)?religião",
        r"guerra\s+(?:de\s+)?religião", 
        r"conflito\s+religioso",
        r"diversidade\s+religiosa",
        r"política\s+(?:e|ou)\s+religião",
        r"estado\s+(?:e|ou)\s+religião",
        r"ensino\s+religioso",
        r"matrimônio\s+religioso"
    ]
    
    for exclusao in contextos_excluir:
        if re.search(exclusao, texto, re.IGNORECASE):
            return False, None
    
    # Busca por padrões muito específicos
    for regex in frases_chave_compiladas:
        match = regex.search(texto)
        if match:
            return True, match.group(0)
    return False, None

def classifica_com_llama_rapida(texto_pdf, tentativa=1):
    """Classificação muito mais rigorosa para reduzir falsos positivos"""
    
    # Prompt ultra rigoroso e específico
    prompt = f"""Analise o questionário abaixo. Seu objetivo é buscar se há ou não alguma pergunta sobre a religião do entrevistado. As perguntas podem ter diferentes formatos de pergunta e resposta.

Responda SIM sempre que encontrar estes tipos ou similares de perguntas, e NÃO caso contrário. Seja extremamente rigoroso e não aceite ambiguidades:

✅ PERGUNTAS DIRETAS VÁLIDAS:
- "Qual é sua religião?" ou "Qual sua religião?"
- "Você tem alguma religião?" 
- "Qual sua denominação religiosa?"
- "Você se considera: ( ) Católico ( ) Evangélico ( ) Protestante"
- "Frequenta algum local religioso?" ou "Frequenta igreja/templo/culto?"
- "Com que frequência vai à missa/culto?"
- "Religião do entrevistado: ( ) Católica ( ) Evangélica"
- "Vou listar algumas religiões"


IMPORTANTE: Se não tiver CERTEZA ABSOLUTA de que há pergunta sobre religião PESSOAL do entrevistado, responda NÃO.

Responda APENAS: SIM ou NÃO

TEXTO:
{texto_pdf[:TEXTO_LIMITE]}

RESPOSTA:"""

    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.01,  # Muito baixa para consistência
                "top_p": 0.1,         # Muito baixa para evitar criatividade
                "num_predict": 3
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
        
        if response.status_code == 200:
            result = response.json()
            resposta = result.get('response', '').strip().upper()
            
            # Análise muito rigorosa da resposta
            if resposta == "SIM":
                return True
            elif "NÃO" in resposta or "NAO" in resposta or resposta == "NO":
                return False
            else:
                # Se resposta ambígua, assume NÃO (conservador)
                logger.warning(f"Resposta ambígua: '{resposta}' - assumindo NÃO")
                return False
        else:
            return None
            
    except Exception:
        return None

def processar_pdf_individual(args):
    """Processa um único PDF - otimizado para threading"""
    idx, row, pdf_dir = args
    pdf_id = str(row["Número da pesquisa"]).zfill(5)
    pdf_path = os.path.join(pdf_dir, f"quest_{pdf_id}.pdf")
    
    resultado = {
        'idx': idx,
        'pdf_id': pdf_id,
        'sucesso': False,
        'contem_religiao': False,
        'metodo': 'erro'
    }
    
    # Verifica se o arquivo existe
    if not os.path.exists(pdf_path):
        resultado['metodo'] = 'arquivo_não_encontrado'
        return resultado
    
    # Extrai texto (com cache)
    texto = ler_pdf_texto_cached(pdf_path)
    if not texto:
        resultado['metodo'] = 'erro_extração'
        return resultado
    
    # Busca por palavras-chave primeiro (método mais confiável e rápido)
    encontrou_palavra, palavra_encontrada = buscar_palavras_chave_otimizada(texto)
    
    if encontrou_palavra:
        resultado.update({
            'sucesso': True,
            'contem_religiao': True,
            'metodo': f'regex: {palavra_encontrada[:50]}'
        })
        return resultado
    
    # Se não encontrou palavra-chave, usa IA (apenas 1 tentativa para velocidade)
    ia_resultado = classifica_com_llama_rapida(texto, 1)
    
    if ia_resultado is not None:
        resultado.update({
            'sucesso': True,
            'contem_religiao': ia_resultado,
            'metodo': 'IA_rapida'
        })
    else:
        resultado['metodo'] = 'IA_timeout'
    
    return resultado

def processar_batch_paralelo(df, inicio_idx, fim_idx):
    """Processa um lote de PDFs em paralelo"""
    logger.info(f"📦 Processando batch {inicio_idx}-{fim_idx} com {WORKERS} threads")
    batch_start = time.time()
    sucessos = 0
    
    # Prepara argumentos para processamento paralelo
    args_list = []
    for idx in range(inicio_idx, min(fim_idx, len(df))):
        if idx <= progresso["ultimo_processado"]:
            continue
        row = df.iloc[idx]
        args_list.append((idx, row, pdf_dir))
    
    if not args_list:
        return 0
    
    # Processa em paralelo
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_idx = {executor.submit(processar_pdf_individual, args): args[0] for args in args_list}
        
        for future in as_completed(future_to_idx):
            try:
                resultado = future.result()
                idx = resultado['idx']
                
                if resultado['sucesso']:
                    sucessos += 1
                    df.at[idx, "contém_religião"] = resultado['contem_religiao']
                    df.at[idx, "método_detecção"] = resultado['metodo']
                    
                    if resultado['contem_religiao']:
                        progresso["total_com_religiao"] += 1
                    
                    status = "SIM" if resultado['contem_religiao'] else "NÃO"
                    if status == "SIM":
                        logger.info(f"✅ PDF {resultado['pdf_id']} - {resultado['metodo']}: {status}")
                    else:
                        logger.debug(f"➖ PDF {resultado['pdf_id']} - {resultado['metodo']}: {status}")
                else:
                    df.at[idx, "contém_religião"] = False
                    df.at[idx, "método_detecção"] = resultado['metodo']
                    logger.debug(f"❌ PDF {resultado['pdf_id']} - {resultado['metodo']}")
                
                progresso["ultimo_processado"] = max(progresso["ultimo_processado"], idx)
                progresso["total_processados"] += 1
            except Exception as e:
                logger.error(f"Erro ao processar resultado: {e}")
                continue
    
    batch_time = time.time() - batch_start
    progresso["batches_completados"] += 1
    taxa = len(args_list) / batch_time if batch_time > 0 else 0
    
    logger.info(f"📦 Batch concluído: {sucessos}/{len(args_list)} sucessos em {batch_time:.2f}s ({taxa:.1f} PDFs/s)")
    return sucessos

# ========= EXECUÇÃO PRINCIPAL =========
def main():
    # Configura manipulador de sinal
    signal.signal(signal.SIGINT, signal_handler)
    
    # Verifica se usuário quer resetar via argumento
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        reset_progresso()
        logger.info("🔄 Progresso resetado. Execute novamente sem --reset para iniciar.")
        return
    
    # Verifica se existe progresso anterior e pergunta se quer resetar
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                prog_anterior = json.load(f)
            
            print(f"\n📋 Progresso anterior encontrado:")
            print(f"   • PDFs processados: {prog_anterior.get('total_processados', 0)}")
            print(f"   • Com religião: {prog_anterior.get('total_com_religiao', 0)}")
            print(f"   • Último processado: {prog_anterior.get('ultimo_processado', -1)}")
            
            resposta = input("\n❓ Deseja CONTINUAR de onde parou (c) ou RESETAR tudo (r)? [c/r]: ").lower().strip()
            
            if resposta == 'r' or resposta == 'reset':
                reset_progresso()
                logger.info("🔄 Progresso resetado! Iniciando do zero...")
            elif resposta == 'c' or resposta == 'continuar' or resposta == '':
                logger.info("▶️  Continuando de onde parou...")
            else:
                logger.info("❌ Opção inválida. Saindo...")
                return
        except:
            logger.warning("⚠️  Erro ao ler progresso anterior. Iniciando do zero...")
    
    logger.info("🔍 Iniciando análise otimizada de PDFs...")
    
    # Verifica se o Ollama está disponível
    if not verificar_ollama():
        logger.error("❌ Ollama não está disponível. Verifique a instalação.")
        return
    
    # Carrega progresso anterior
    carregar_progresso()
    
    # Carrega o CSV
    try:
        df = pd.read_csv(csv_path)
        if "contém_religião" not in df.columns:
            df["contém_religião"] = False
        if "método_detecção" not in df.columns:
            df["método_detecção"] = ""
    except FileNotFoundError:
        logger.error(f"❌ Arquivo CSV não encontrado: {csv_path}")
        return
    
    total_pdfs = len(df)
    logger.info(f"📊 Total de PDFs para processar: {total_pdfs}")
    
    if progresso["inicio_processamento"] is None:
        progresso["inicio_processamento"] = datetime.now().isoformat()
    
    start_time = time.time()
    
    # Processa em batches
    for batch_inicio in range(progresso["ultimo_processado"] + 1, total_pdfs, BATCH_SIZE):
        batch_fim = min(batch_inicio + BATCH_SIZE, total_pdfs)
        
        try:
            sucessos = processar_batch_paralelo(df, batch_inicio, batch_fim)
            
            # Salva progresso após cada batch (menos frequente para velocidade)
            salvar_progresso()
            df.to_csv(output_csv_path, index=False)
            
            # Estatísticas com taxa de processamento
            porcentagem = (progresso["total_processados"] / total_pdfs) * 100
            tempo_decorrido = time.time() - start_time
            if progresso["total_processados"] > 0:
                tempo_estimado = (tempo_decorrido / progresso["total_processados"]) * (total_pdfs - progresso["total_processados"])
                taxa_atual = progresso["total_processados"] / tempo_decorrido
            else:
                tempo_estimado = 0
                taxa_atual = 0
            
            logger.info(f"📈 Progresso: {progresso['total_processados']}/{total_pdfs} ({porcentagem:.1f}%)")
            logger.info(f"⚡ Taxa: {taxa_atual:.1f} PDFs/s | ⏱️ Tempo restante: {tempo_estimado/60:.1f} min")
            logger.info(f"✅ Com religião: {progresso['total_com_religiao']} ({progresso['total_com_religiao']/max(progresso['total_processados'],1)*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"❌ Erro no batch {batch_inicio}-{batch_fim}: {e}")
            salvar_progresso()
            continue
    
    # Finalização
    tempo_total = time.time() - start_time
    logger.info(f"🎉 Análise concluída!")
    logger.info(f"📊 Total processados: {progresso['total_processados']}")
    logger.info(f"✅ Com religião: {progresso['total_com_religiao']}")
    logger.info(f"⏱️  Tempo total: {tempo_total/60:.1f} minutos")
    
    # Salva resultado final
    df.to_csv(output_csv_path, index=False)
    logger.info(f"💾 Resultado salvo em: {output_csv_path}")
    
    # Remove arquivo de progresso
    if os.path.exists(progress_file):
        os.remove(progress_file)

if __name__ == "__main__":
    main()