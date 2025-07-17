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

# ========= CONFIGURAÇÕES =========
# Caminhos
pdf_dir = "./questionarios"
csv_path = "pesquisas_cesop.csv"
output_csv_path = "pesquisas_cesop_com_religiao.csv"
progress_file = "progresso_analise.json"
log_file = "analise_pdfs.log"

# Configurações do Llama/Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

# Configurações de processamento
BATCH_SIZE = 10
TIMEOUT_SECONDS = 45  # Aumentado para dar mais tempo para análise
MAX_RETRIES = 3
TEXTO_LIMITE = 20000  # Aumentado para capturar mais contexto

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

# ========= FRASES DE BUSCA MELHORADAS =========
# Padrões mais específicos e rigorosos
frases_chave_especificas = [
    # Perguntas diretas sobre religião
    r"qual\s+(?:é\s+)?(?:a\s+)?sua\s+religião",
    r"você\s+tem\s+alguma\s+religião",
    r"qual\s+(?:a\s+)?sua\s+denominação\s+religiosa",
    r"você\s+se\s+considera\s+(?:católico|evangélico|protestante|espírita)",
    r"frequenta\s+(?:algum\s+)?(?:local|lugar)\s+religioso",
    r"frequenta\s+(?:cultos?|missas?|serviços?\s+religiosos?)",
    r"pratica\s+alguma\s+(?:fé|religião)",
    r"tem\s+alguma\s+(?:fé|crença)\s+religiosa",
    
    # Denominações específicas em contexto de pergunta
    r"(?:católico|evangélico|protestante|espírita|budista|judaico|islâmico|umbanda|candomblé)\s*[?:]",
    r"assembleia\s+de\s+deus",
    r"igreja\s+universal\s+do\s+reino\s+de\s+deus",
    r"testemunhas?\s+de\s+jeová",
    
    # Padrões de questionário
    r"religião\s+do\s+entrevistado",
    r"denominação\s+religiosa\s+do\s+entrevistado",
    r"filiação\s+religiosa",
    r"orientação\s+religiosa",
]

# Palavras que podem gerar falsos positivos (para exclusão)
palavras_exclusao = [
    "religiosamente",  # advérbio
    "religioso" + r"\s+(?:da|do|na|no)",  # contexto não-pergunta
    "política" + r"\s+(?:e|ou)\s+religião",  # contexto acadêmico
    "estado" + r"\s+(?:e|ou)\s+religião",  # contexto acadêmico
    "guerra" + r"\s+(?:de|da)\s+religião",  # contexto histórico
    "liberdade" + r"\s+(?:de|da)\s+religião",  # contexto jurídico
]

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

def ler_pdf_texto(pdf_path):
    """Extrai texto do PDF com tratamento de erros"""
    try:
        reader = PdfReader(pdf_path)
        texto = ""
        for page_num, page in enumerate(reader.pages):
            try:
                texto += page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Erro na página {page_num} do PDF {pdf_path}: {e}")
                continue
        return texto.lower()
    except Exception as e:
        logger.error(f"Erro lendo PDF {pdf_path}: {e}")
        return ""

def buscar_palavras_chave_melhorada(texto):
    """Busca melhorada por palavras-chave usando regex"""
    # Primeiro verifica se há palavras de exclusão
    for exclusao in palavras_exclusao:
        if re.search(exclusao, texto, re.IGNORECASE):
            return False, None
    
    # Busca por padrões específicos
    for padrao in frases_chave_especificas:
        match = re.search(padrao, texto, re.IGNORECASE)
        if match:
            return True, match.group(0)
    
    return False, None

def extrair_secao_questionario(texto):
    """Extrai apenas a seção de perguntas do questionário"""
    # Padrões que indicam início de perguntas
    padroes_inicio = [
        r"quest(?:ões|ionário)",
        r"perguntas?",
        r"entrevista",
        r"^(?:\d+[\.\)]|\w[\.\)])",  # numeração de perguntas
        r"agora\s+(?:eu\s+)?(?:vou\s+)?(?:fazer|perguntar)",
    ]
    
    # Padrões que indicam fim de perguntas
    padroes_fim = [
        r"(?:fim|final)\s+(?:do\s+)?(?:questionário|da\s+entrevista)",
        r"obrigad[oa]",
        r"agradecemos",
        r"referências?\s+bibliográficas?",
    ]
    
    # Procura pelo início da seção de perguntas
    inicio = 0
    for padrao in padroes_inicio:
        match = re.search(padrao, texto, re.IGNORECASE)
        if match:
            inicio = match.start()
            break
    
    # Procura pelo fim da seção de perguntas
    fim = len(texto)
    for padrao in padroes_fim:
        match = re.search(padrao, texto[inicio:], re.IGNORECASE)
        if match:
            fim = inicio + match.start()
            break
    
    return texto[inicio:fim]

def classifica_com_llama_melhorada(texto_pdf, tentativa=1):
    """Classifica usando Llama com prompt melhorado"""
    
    # Extrai apenas a seção relevante
    secao_questionario = extrair_secao_questionario(texto_pdf)
    
    # Prompt muito mais específico e rigoroso
    prompt = f"""Você é um especialista em análise de questionários de pesquisa política e social.

INSTRUÇÕES CRÍTICAS:
1. Analise APENAS se existem PERGUNTAS DIRETAS sobre religião, denominação religiosa ou práticas religiosas
2. NÃO considere menções casuais, acadêmicas ou contextuais sobre religião
3. NÃO considere referências a "liberdade religiosa", "conflitos religiosos" ou "política e religião"
4. PROCURE especificamente por perguntas como:
   - "Qual é sua religião?"
   - "Você tem alguma religião?"
   - "Frequenta algum local religioso?"
   - "Qual sua denominação religiosa?"
   - "Você se considera católico/evangélico/etc?"

EXEMPLOS DE O QUE NÃO CONTA:
- "O candidato respeita a diversidade religiosa" (opinião política)
- "Conflitos entre estado e religião" (contexto acadêmico)
- "Liberdade religiosa" (direito constitucional)
- "Guerra religiosa" (contexto histórico)

EXEMPLOS DE O QUE CONTA:
- "Qual é sua religião?" (pergunta direta)
- "Você frequenta cultos?" (pergunta sobre prática)
- "Católico ( ) Evangélico ( )" (opções de resposta)

IMPORTANTE: Se não encontrar perguntas DIRETAS sobre religião pessoal do entrevistado, responda "NÃO".

RESPONDA APENAS: "SIM" ou "NÃO" (sem explicações)

TEXTO PARA ANÁLISE:
{secao_questionario[:TEXTO_LIMITE]}

RESPOSTA:"""

    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.05,  # Muito baixa para respostas consistentes
                "top_p": 0.1,        # Muito baixa para evitar criatividade
                "num_predict": 10,   # Aumentado para capturar resposta completa
                "stop": ["\n", ".", "Explicação", "Porque"]  # Para parar explicações
            }
        }
        
        response = requests.post(
            OLLAMA_URL, 
            json=payload, 
            timeout=TIMEOUT_SECONDS
        )
        
        if response.status_code == 200:
            result = response.json()
            resposta = result.get('response', '').strip().upper()
            
            # Análise mais rigorosa da resposta
            if "SIM" in resposta and "NÃO" not in resposta:
                return True
            elif "NÃO" in resposta or "NAO" in resposta:
                return False
            else:
                logger.warning(f"Resposta ambígua da IA: {resposta}")
                return False  # Em caso de dúvida, assume NÃO
        else:
            logger.warning(f"Erro HTTP {response.status_code} na tentativa {tentativa}")
            return None
            
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout na tentativa {tentativa}")
        return None
    except Exception as e:
        logger.warning(f"Erro na tentativa {tentativa}: {e}")
        return None

def validacao_dupla_ia(texto_pdf):
    """Faz validação dupla com prompts diferentes para reduzir falsos positivos"""
    
    # Primeiro prompt: mais permissivo
    resultado1 = classifica_com_llama_melhorada(texto_pdf, 1)
    
    if resultado1 is False:
        return False  # Se o primeiro já disse NÃO, não precisa do segundo
    
    if resultado1 is True:
        # Segundo prompt: mais rigoroso para validar
        prompt_validacao = f"""Você é um revisor rigoroso de análises de questionários.

O texto abaixo foi classificado como "contendo perguntas sobre religião".
Sua tarefa é VALIDAR se isso está CORRETO.

SEJA EXTREMAMENTE RIGOROSO:
- Procure por perguntas DIRETAS sobre religião pessoal
- NÃO aceite menções casuais, acadêmicas ou políticas
- NÃO aceite contextos sobre "liberdade religiosa" ou "conflitos religiosos"

Se tiver QUALQUER dúvida, responda "NÃO".

RESPONDA APENAS: "SIM" ou "NÃO"

TEXTO:
{texto_pdf[:TEXTO_LIMITE]}

VALIDAÇÃO:"""

        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt_validacao,
                "stream": False,
                "options": {
                    "temperature": 0.01,
                    "top_p": 0.05,
                    "num_predict": 5,
                    "stop": ["\n", "."]
                }
            }
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
            
            if response.status_code == 200:
                result = response.json()
                resposta = result.get('response', '').strip().upper()
                
                if "SIM" in resposta and "NÃO" not in resposta:
                    return True
                else:
                    progresso["falsos_positivos_evitados"] += 1
                    return False
            else:
                return False  # Em caso de erro, assume NÃO
        except:
            return False  # Em caso de erro, assume NÃO
    
    return None  # Se primeiro resultado foi None

def processar_pdf(idx, row, df):
    """Processa um único PDF com validação melhorada"""
    pdf_id = str(row["Número da pesquisa"]).zfill(5)
    pdf_path = os.path.join(pdf_dir, f"quest_{pdf_id}.pdf")
    
    # Verifica se o arquivo existe
    if not os.path.exists(pdf_path):
        logger.warning(f"⚠️  PDF não encontrado: {pdf_path}")
        return False
    
    # Extrai texto
    texto = ler_pdf_texto(pdf_path)
    if not texto:
        logger.warning(f"⚠️  Não foi possível extrair texto do PDF: {pdf_id}")
        return False
    
    # Busca por palavras-chave primeiro (método mais confiável)
    encontrou_palavra, palavra_encontrada = buscar_palavras_chave_melhorada(texto)
    
    if encontrou_palavra:
        df.at[idx, "contém_religião"] = True
        df.at[idx, "método_detecção"] = f"regex: {palavra_encontrada}"
        logger.info(f"✅ PDF {pdf_id} - Regex encontrou: {palavra_encontrada}")
        return True
    
    # Se não encontrou palavra-chave, usa IA com validação dupla
    logger.info(f"🤖 PDF {pdf_id} - Analisando com IA...")
    
    for tentativa in range(1, MAX_RETRIES + 1):
        resultado = validacao_dupla_ia(texto)
        
        if resultado is not None:
            df.at[idx, "contém_religião"] = resultado
            df.at[idx, "método_detecção"] = f"IA_dupla (tentativa {tentativa})"
            status = "SIM" if resultado else "NÃO"
            logger.info(f"✅ PDF {pdf_id} - IA dupla: {status}")
            return True
        
        if tentativa < MAX_RETRIES:
            logger.warning(f"⏳ PDF {pdf_id} - Tentativa {tentativa} falhou, tentando novamente...")
            time.sleep(3)
    
    # Se chegou aqui, todas as tentativas falharam
    logger.error(f"❌ PDF {pdf_id} - Todas as tentativas falharam")
    df.at[idx, "contém_religião"] = False
    df.at[idx, "método_detecção"] = "falha_na_análise"
    return False

def processar_batch(df, inicio_idx, fim_idx):
    """Processa um lote de PDFs"""
    logger.info(f"📦 Processando batch {inicio_idx}-{fim_idx}")
    batch_start = time.time()
    sucessos = 0
    
    for idx in range(inicio_idx, min(fim_idx, len(df))):
        if idx <= progresso["ultimo_processado"]:
            continue
            
        row = df.iloc[idx]
        sucesso = processar_pdf(idx, row, df)
        
        if sucesso:
            sucessos += 1
            if df.at[idx, "contém_religião"]:
                progresso["total_com_religiao"] += 1
        
        progresso["ultimo_processado"] = idx
        progresso["total_processados"] += 1
        
        # Salva progresso a cada 5 PDFs
        if (idx + 1) % 5 == 0:
            salvar_progresso()
    
    batch_time = time.time() - batch_start
    progresso["batches_completados"] += 1
    
    logger.info(f"📦 Batch concluído: {sucessos}/{fim_idx-inicio_idx} sucessos em {batch_time:.2f}s")
    return sucessos

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
    
    logger.info("🔍 Iniciando análise melhorada de PDFs...")
    
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
            sucessos = processar_batch(df, batch_inicio, batch_fim)
            
            # Salva progresso após cada batch
            salvar_progresso()
            df.to_csv(output_csv_path, index=False)
            
            # Estatísticas
            porcentagem = (progresso["total_processados"] / total_pdfs) * 100
            tempo_decorrido = time.time() - start_time
            tempo_estimado = (tempo_decorrido / progresso["total_processados"]) * (total_pdfs - progresso["total_processados"])
            
            logger.info(f"📈 Progresso: {progresso['total_processados']}/{total_pdfs} ({porcentagem:.1f}%)")
            logger.info(f"⏱️  Tempo estimado restante: {tempo_estimado/60:.1f} minutos")
            logger.info(f"🛡️  Falsos positivos evitados: {progresso['falsos_positivos_evitados']}")
            
        except Exception as e:
            logger.error(f"❌ Erro no batch {batch_inicio}-{batch_fim}: {e}")
            salvar_progresso()
            continue
    
    # Finalização
    tempo_total = time.time() - start_time
    logger.info(f"🎉 Análise concluída!")
    logger.info(f"📊 Total processados: {progresso['total_processados']}")
    logger.info(f"✅ Com religião: {progresso['total_com_religiao']}")
    logger.info(f"🛡️  Falsos positivos evitados: {progresso['falsos_positivos_evitados']}")
    logger.info(f"⏱️  Tempo total: {tempo_total/60:.1f} minutos")
    
    # Salva resultado final
    df.to_csv(output_csv_path, index=False)
    logger.info(f"💾 Resultado salvo em: {output_csv_path}")
    
    # Remove arquivo de progresso
    if os.path.exists(progress_file):
        os.remove(progress_file)

if __name__ == "__main__":
    main()