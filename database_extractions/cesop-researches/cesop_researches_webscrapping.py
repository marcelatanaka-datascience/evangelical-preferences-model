import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# ========== CONFIGURAÇÕES ==========
TOTAL_PAGINAS = 458
BLOCO_PAGINAS = 20
CSV_FINAL = "pesquisas_cesop.csv"
CHECKPOINT = "checkpoint_parcial.csv"

# ========== FUNÇÕES ==========

def iniciar_driver():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Comente esta linha se quiser ver o navegador
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.page_load_strategy = 'eager'
    return webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)

def limpar_valor(valor_com_prefixo):
    if isinstance(valor_com_prefixo, str) and ":\n" in valor_com_prefixo:
        return valor_com_prefixo.split(":\n", 1)[1].strip()
    return valor_com_prefixo.strip() if isinstance(valor_com_prefixo, str) else valor_com_prefixo

def salvar_checkpoint(dados, arquivo):
    df = pd.DataFrame(dados)
    df.to_csv(arquivo, index=False, encoding="utf-8-sig")
    print(f"💾 Checkpoint salvo: {arquivo}")

def carregar_checkpoint():
    if os.path.exists(CHECKPOINT):
        print("📂 Checkpoint detectado. Retomando extração...")
        df = pd.read_csv(CHECKPOINT)
        return df.to_dict(orient='records'), (len(df) // 10) + 1
    return [], 1

# ========== EXECUÇÃO PRINCIPAL ==========

dados, pagina_inicial = carregar_checkpoint()
driver = None
BASE_URL = "https://www.cesop.unicamp.br/por/banco_de_dados/page:{page}?ext=html"

for bloco_inicio in range(pagina_inicial, TOTAL_PAGINAS + 1, BLOCO_PAGINAS):
    bloco_fim = min(bloco_inicio + BLOCO_PAGINAS - 1, TOTAL_PAGINAS)
    print(f"\n🚀 Iniciando bloco {bloco_inicio}-{bloco_fim}")

    try:
        driver = iniciar_driver()

        for page in range(bloco_inicio, bloco_fim + 1):
            try:
                url = BASE_URL.format(page=page)
                print(f"🔄 Página {page}/{TOTAL_PAGINAS} - {url}")
                driver.set_page_load_timeout(60)
                driver.get(url)
                time.sleep(2.5)

                linhas = driver.find_elements(By.CSS_SELECTOR, ".line-articles.row-data")
                for linha in linhas:
                    colunas = linha.find_elements(By.CLASS_NAME, "col-data")
                    if len(colunas) < 5:
                        continue

                    numero = limpar_valor(colunas[0].text)
                    titulo_tag = colunas[1].find_element(By.TAG_NAME, "a")
                    titulo = limpar_valor(titulo_tag.text)
                    link = titulo_tag.get_attribute("href")
                    data = limpar_valor(colunas[2].text)
                    local = limpar_valor(colunas[3].text)
                    instituto = limpar_valor(colunas[4].text)

                    dados.append({
                        'Número da pesquisa': numero,
                        'Título': titulo,
                        'Link da pesquisa': link,
                        'Data': data,
                        'Local': local,
                        'Instituto de pesquisa': instituto
                    })
                print(f"✅ Página {page} extraída com sucesso")

            except (TimeoutException, WebDriverException) as e:
                print(f"⚠️ Erro ao carregar página {page}: {str(e)[:200]}")
                time.sleep(5)
                continue

        salvar_checkpoint(dados, CHECKPOINT)
        driver.quit()
        driver = None

    except Exception as e:
        print(f"❌ Erro inesperado no bloco {bloco_inicio}-{bloco_fim}: {str(e)}")
        if driver:
            driver.quit()
        break

# Salvar CSV final
df_final = pd.DataFrame(dados)
df_final.to_csv(CSV_FINAL, index=False, encoding='utf-8-sig')
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)

print("✅ Extração e limpeza concluídas com sucesso. Arquivo salvo:", CSV_FINAL)
