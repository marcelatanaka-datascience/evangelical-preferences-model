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
BATCH_SIZE = 30
TIMEOUT_SECONDS = 15
MAX_RETRIES = 2
TEXTO_LIMITE = 20000
WORKERS = 4

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

# ========= REGEX SIMPLES DO CODIGO 1 =========
# Regex simples e eficaz para "religião" e suas variantes
religiao_regex = re.compile(r"religi[aã]o", re.IGNORECASE)

# Cache de textos já processados
texto_cache = {}
cache_lock = threading.Lock()

# ========= CONTROLE DE PARADA =========
parar_processamento = threading.Event()
executor_global = None

# ========= VARIÁVEIS GLOBAIS =========
progresso = {
    "ultimo_processado": -1,
    "total_com_religiao": 0,
    "total_processados": 0,
    "inicio_processamento": None,
    "batches_completados": 0,
    "regex_encontrou": 0,
    "ia_encontrou": 0,
    "ia_nao_encontrou": 0
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
    logger.info("🛑 Interrupção detectada! Parando processamento...")
    parar_processamento.set()
    
    # Para o executor se estiver rodando
    global executor_global
    if executor_global:
        logger.info("⏹️  Cancelando tarefas pendentes...")
        executor_global.shutdown(wait=False, cancel_futures=True)
    
    # Salva progresso atual
    logger.info("💾 Salvando progresso atual...")
    salvar_progresso()
    
    logger.info("✅ Parada segura concluída!")
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
        "regex_encontrou": 0,
        "ia_encontrou": 0,
        "ia_nao_encontrou": 0
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
        # Lê todas as páginas para garantir busca completa
        for page in reader.pages:
            try:
                texto += page.extract_text() or ""
            except Exception:
                continue
        
        with cache_lock:
            texto_cache[pdf_path] = texto
            # Limita cache a 1000 entradas
            if len(texto_cache) > 1000:
                oldest_key = next(iter(texto_cache))
                del texto_cache[oldest_key]
        
        return texto
    except Exception as e:
        logger.warning(f"Erro lendo PDF {pdf_path}: {e}")
        return ""

def buscar_religiao_regex(texto):
    """Busca usando o regex simples do CODIGO 1"""
    return bool(religiao_regex.search(texto))

def classifica_com_llama_rapida(texto_pdf, tentativa=1):
    """Classificação com IA apenas quando regex não encontra nada"""
    
    prompt = f"""Analise o questionário abaixo. Seu objetivo é buscar se há ou não alguma pergunta sobre a religião do entrevistado.

Responda SIM sempre que encontrar perguntas sobre religião pessoal do entrevistado, como:
- "Qual é sua religião?" ou "Qual sua religião?"
- "Você tem alguma religião?" 
- "Qual sua denominação religiosa?"
- "Você se considera: ( ) Católico ( ) Evangélico ( ) Protestante"
- "Frequenta algum local religioso?" ou "Frequenta igreja/templo/culto?"
- "Com que frequência vai à missa/culto?"
- "Religião do entrevistado: ( ) Católica ( ) Evangélica"

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
                "temperature": 0.01,
                "top_p": 0.1,
                "num_predict": 3
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
        
        if response.status_code == 200:
            result = response.json()
            resposta = result.get('response', '').strip().upper()
            
            if resposta == "SIM":
                return True
            elif "NÃO" in resposta or "NAO" in resposta or resposta == "NO":
                return False
            else:
                logger.warning(f"Resposta ambígua: '{resposta}' - assumindo NÃO")
                return False
        else:
            return None
            
    except Exception:
        return None

def processar_pdf_individual(args):
    """Processa um único PDF - primeiro regex, depois IA se necessário"""
    # Verifica se deve parar
    if parar_processamento.is_set():
        return None
        
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
    
    # Verifica novamente se deve parar
    if parar_processamento.is_set():
        return None
    
    # Verifica se o arquivo existe
    if not os.path.exists(pdf_path):
        resultado['metodo'] = 'arquivo_não_encontrado'
        logger.warning(f"[{pdf_id}] ❌ Arquivo não encontrado")
        return resultado
    
    # Extrai texto (com cache)
    texto = ler_pdf_texto_cached(pdf_path)
    if not texto:
        resultado['metodo'] = 'erro_extração'
        logger.warning(f"[{pdf_id}] ❌ Erro na extração de texto")
        return resultado
    
    # Verifica se deve parar antes do regex
    if parar_processamento.is_set():
        return None
    
    # PASSO 1: Busca com regex simples (método do CODIGO 1)
    encontrou_regex = buscar_religiao_regex(texto)
    
    if encontrou_regex:
        resultado.update({
            'sucesso': True,
            'contem_religiao': True,
            'metodo': 'regex'
        })
        logger.info(f"[{pdf_id}] ✅ REGEX encontrou - SIM")
        progresso["regex_encontrou"] += 1
        return resultado
    
    # Verifica se deve parar antes da IA
    if parar_processamento.is_set():
        return None
    
    # PASSO 2: Se regex não encontrou, usa IA
    logger.info(f"[{pdf_id}] 🔍 REGEX não encontrou, consultando IA...")
    ia_resultado = classifica_com_llama_rapida(texto, 1)
    
    # Verifica se deve parar após a IA
    if parar_processamento.is_set():
        return None
    
    if ia_resultado is not None:
        resultado.update({
            'sucesso': True,
            'contem_religiao': ia_resultado,
            'metodo': 'IA'
        })
        
        if ia_resultado:
            logger.info(f"[{pdf_id}] ✅ IA encontrou - SIM")
            progresso["ia_encontrou"] += 1
        else:
            logger.info(f"[{pdf_id}] ❌ IA não encontrou - NÃO")
            progresso["ia_nao_encontrou"] += 1
    else:
        resultado.update({
            'sucesso': True,
            'contem_religiao': False,
            'metodo': 'IA_timeout'
        })
        logger.warning(f"[{pdf_id}] ⏰ IA timeout - assumindo NÃO")
    
    return resultado

def processar_batch_paralelo(df, inicio_idx, fim_idx):
    """Processa um lote de PDFs em paralelo"""
    global executor_global
    
    # Verifica se deve parar antes de começar o batch
    if parar_processamento.is_set():
        logger.info("🛑 Parando batch devido à interrupção")
        return 0
        
    logger.info(f"📦 Processando batch {inicio_idx}-{fim_idx} com {WORKERS} threads")
    batch_start = time.time()
    sucessos = 0
    
    # Prepara argumentos para processamento paralelo
    args_list = []
    for idx in range(inicio_idx, min(fim_idx, len(df))):
        if idx <= progresso["ultimo_processado"]:
            continue
        if parar_processamento.is_set():
            break
        row = df.iloc[idx]
        args_list.append((idx, row, pdf_dir))
    
    if not args_list:
        return 0
    
    # Processa em paralelo com controle de parada
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            executor_global = executor  # Referência global para poder cancelar
            
            # Submete todas as tarefas
            future_to_idx = {}
            for args in args_list:
                if parar_processamento.is_set():
                    break
                future = executor.submit(processar_pdf_individual, args)
                future_to_idx[future] = args[0]
            
            # Processa resultados conforme ficam prontos
            for future in as_completed(future_to_idx):
                if parar_processamento.is_set():
                    logger.info("🛑 Parando processamento de resultados")
                    # Cancela futures pendentes
                    for f in future_to_idx:
                        if not f.done():
                            f.cancel()
                    break
                    
                try:
                    resultado = future.result(timeout=1)  # Timeout curto para responsividade
                    
                    # Se retornou None, foi interrompido
                    if resultado is None:
                        continue
                        
                    idx = resultado['idx']
                    
                    if resultado['sucesso']:
                        sucessos += 1
                        df.at[idx, "contém_religião"] = resultado['contem_religiao']
                        df.at[idx, "método_detecção"] = resultado['metodo']
                        
                        if resultado['contem_religiao']:
                            progresso["total_com_religiao"] += 1
                    else:
                        df.at[idx, "contém_religião"] = False
                        df.at[idx, "método_detecção"] = resultado['metodo']
                    
                    progresso["ultimo_processado"] = max(progresso["ultimo_processado"], idx)
                    progresso["total_processados"] += 1
                    
                except Exception as e:
                    if not parar_processamento.is_set():
                        logger.error(f"Erro ao processar resultado: {e}")
                    continue
            
            executor_global = None  # Limpa referência
            
    except Exception as e:
        if not parar_processamento.is_set():
            logger.error(f"Erro no executor: {e}")
    
    if parar_processamento.is_set():
        logger.info("🛑 Batch interrompido pelo usuário")
        return sucessos
    
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
            print(f"   • Regex encontrou: {prog_anterior.get('regex_encontrou', 0)}")
            print(f"   • IA encontrou: {prog_anterior.get('ia_encontrou', 0)}")
            
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
    
    logger.info("🔍 Iniciando análise otimizada: REGEX primeiro, depois IA")
    
    # Verifica se o Ollama está disponível (para quando precisar da IA)
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
        # Verifica se deve parar antes de cada batch
        if parar_processamento.is_set():
            logger.info("🛑 Processamento interrompido pelo usuário")
            break
            
        batch_fim = min(batch_inicio + BATCH_SIZE, total_pdfs)
        
        try:
            sucessos = processar_batch_paralelo(df, batch_inicio, batch_fim)
            
            # Verifica se deve parar após o batch
            if parar_processamento.is_set():
                logger.info("🛑 Salvando progresso antes de parar...")
                salvar_progresso()
                df.to_csv(output_csv_path, index=False)
                logger.info("💾 Progresso salvo com sucesso!")
                break
            
            # Salva progresso após cada batch
            salvar_progresso()
            df.to_csv(output_csv_path, index=False)
            
            # Estatísticas detalhadas
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
            logger.info(f"🔍 Regex: {progresso['regex_encontrou']} | 🤖 IA SIM: {progresso['ia_encontrou']} | ❌ IA NÃO: {progresso['ia_nao_encontrou']}")
            
        except Exception as e:
            logger.error(f"❌ Erro no batch {batch_inicio}-{batch_fim}: {e}")
            salvar_progresso()
            continue
    
    # Finalização
    if parar_processamento.is_set():
        logger.info("🛑 Processamento foi interrompido pelo usuário")
        logger.info(f"📊 Processados até agora: {progresso['total_processados']}")
        logger.info(f"✅ Com religião: {progresso['total_com_religiao']}")
        logger.info(f"💾 Progresso salvo. Execute novamente para continuar.")
        return
    
    tempo_total = time.time() - start_time
    logger.info(f"🎉 Análise concluída!")
    logger.info(f"📊 Total processados: {progresso['total_processados']}")
    logger.info(f"✅ Com religião: {progresso['total_com_religiao']}")
    logger.info(f"🔍 Encontrados por REGEX: {progresso['regex_encontrou']}")
    logger.info(f"🤖 Encontrados por IA: {progresso['ia_encontrou']}")
    logger.info(f"❌ Não encontrados pela IA: {progresso['ia_nao_encontrou']}")
    logger.info(f"⏱️  Tempo total: {tempo_total/60:.1f} minutos")
    
    # Salva resultado final
    df.to_csv(output_csv_path, index=False)
    logger.info(f"💾 Resultado salvo em: {output_csv_path}")
    
    # Remove arquivo de progresso
    if os.path.exists(progress_file):
        os.remove(progress_file)

if __name__ == "__main__":
    main()