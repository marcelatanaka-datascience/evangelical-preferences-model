import os
import sys
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from getpass import getpass

CSV_PATH = '/Users/marcelatanaka/evangelical-preferences-model/database_extractions/cesop-researches/databases_index_full.csv'
OUTPUT_DIR = '/Users/marcelatanaka/evangelical-preferences-model/database_extractions/spss_files'
TIMEOUT = 50  # segundos de espera para ações no navegador
DOWNLOAD_TIMEOUT = 120  # segundos para conclusão do download

# Solicita credenciais de login
print("=== CREDENCIAIS DE LOGIN ===")
USERNAME = input("Digite seu usuário/email: ")
PASSWORD = getpass("Digite sua senha: ")

# Verifica existência do CSV
def verify_csv(path):
    if not os.path.isfile(path):
        sys.exit(f"Erro: arquivo não encontrado em {path}")

verify_csv(CSV_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configura WebDriver Chrome
def init_driver(download_dir):
    chrome_opts = Options()
    # chrome_opts.add_argument('--headless=new')  # descomente para rodar sem UI
    chrome_opts.add_argument('--no-sandbox')
    chrome_opts.add_argument('--disable-dev-shm-usage')
    prefs = {
        'download.default_directory': download_dir,
        'download.prompt_for_download': False,
        'download.directory_upgrade': True,
        'safebrowsing.enabled': True
    }
    chrome_opts.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(options=chrome_opts)
    # Permite downloads em headless
    driver.execute_cdp_cmd('Page.setDownloadBehavior', {
        'behavior': 'allow',
        'downloadPath': download_dir
    })
    return driver

# Realiza login no site
def perform_login(driver, username, password):
    print("🔐 Iniciando login...")
    driver.get('https://www.cesop.unicamp.br/por/cesop/main/login')
    try:
        wait = WebDriverWait(driver, TIMEOUT)
        email = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[name='data[CspUser][email]']")))
        pwd = driver.find_element(By.CSS_SELECTOR, "input[name='data[CspUser][password]']")
        submit = driver.find_element(By.CSS_SELECTOR, "input[type='submit'][value='Login']")
        email.clear(); email.send_keys(username)
        pwd.clear(); pwd.send_keys(password)
        submit.click()
        time.sleep(5)
        if 'login' in driver.current_url.lower():
            print("❌ Falha no login: permaneceu na página de login.")
            return False
        print("✅ Login efetuado com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro durante login: {e}")
        return False

# Localiza o botão SPSS e faz download
def find_and_download_spss(driver, url, download_dir):
    print(f"🔗 Acessando: {url}")
    driver.get(url)
    time.sleep(3)
    try:
        btn = WebDriverWait(driver, TIMEOUT).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(normalize-space(.),'SPSS') and not(contains(@class,'blocked'))]"))
        )
        print("✅ Botão SPSS encontrado, clicando...")
        btn.click()
    except TimeoutException:
        print("❌ Botão SPSS não encontrado ou não clicável.")
        debug_spss_buttons(driver)
        return False
    except Exception as e:
        print(f"⚠️ Erro ao clicar no botão SPSS: {e}")
        debug_spss_buttons(driver)
        return False

    return wait_for_download(download_dir)

# Imprime debug de botões com texto ou href contendo 'spss'
def debug_spss_buttons(driver):
    print("🔍 Debug: listando elementos potencialmente SPSS...")
    elems = driver.find_elements(By.XPATH, "//a[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'spss')]")
    elems += driver.find_elements(By.XPATH, "//a[contains(translate(@href,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'spss')]")
    if not elems:
        print("   (nenhum elemento SPSS encontrado)")
    for idx, el in enumerate(elems[:5], 1):
        txt = el.text.strip()
        cls = el.get_attribute('class')
        href = el.get_attribute('href')
        print(f"   {idx}. Texto='{txt}' | Classes='{cls}' | href='{href}'")

# Aguarda conclusão do download de um novo arquivo .zip

def wait_for_download(download_dir, timeout=DOWNLOAD_TIMEOUT):
    print("⏳ Aguardando download...")
    start = time.time()
    before = set(os.listdir(download_dir))
    while time.time() - start < timeout:
        after = set(os.listdir(download_dir))
        new = after - before
        if new:
            zips = [f for f in new if f.lower().endswith('.zip')]
            if zips:
                print(f"✅ Download concluído: {zips}")
                return True
        time.sleep(1)
    print("⏰ Timeout aguardando download.")
    return False

# Fluxo principal
def main():
    df = pd.read_csv(CSV_PATH)
    print(f"🚀 {len(df)} URLs carregadas do CSV.")
    driver = init_driver(OUTPUT_DIR)
    if not perform_login(driver, USERNAME, PASSWORD):
        driver.quit(); sys.exit(1)
    for i, row in df.iterrows():
        url = row.get('Link da pesquisa')
        if not isinstance(url, str) or not url.startswith('http'):
            print(f"[{i}] URL inválida: {url}")
            continue
        find_and_download_spss(driver, url, OUTPUT_DIR)
        time.sleep(2)
    driver.quit()
    print(f"🏁 Processo finalizado. Arquivos em {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
