import os
import re
import logging
import unicodedata

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
import nltk

# === CONFIGURAÇÃO DO LOGGING ===
logging.basicConfig(
    level=logging.INFO,  # Trocar para DEBUG para mais detalhes
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

logging.info("Iniciando análise do debate.")

# === 1. Carregar Corpus ===
try:
    data_file = os.path.join(os.path.dirname(__file__), "debate_sbt_1989_1T.txt")
    with open(data_file, "r", encoding="utf-8") as f:
        texto = f.read()
    logging.info("Arquivo carregado com sucesso (%s). Tamanho do texto: %d caracteres", data_file, len(texto))
except Exception as e:
    logging.error("Erro ao carregar arquivo: %s", e)
    raise

# === 2. Separar falas ===
logging.info("Separando falas por candidato...")
falas = re.split(r"\n(?=[A-ZÁÂÃÉÊÍÓÔÕÚÇ][A-ZÁÂÃÉÊÍÓÔÕÚÇ\s\-.]{1,40}:)", texto)
data = []
for fala in falas:
    partes = fala.split(":", 1)
    if len(partes) == 2:
        candidato, discurso = partes
        data.append([candidato.strip(), discurso.strip()])

if not data:
    logging.warning("Nenhum rótulo de candidato detectado. Usando fallback 'GERAL'.")
    data = [["GERAL", texto.strip()]]

df = pd.DataFrame(data, columns=["candidato", "fala"])
logging.info("Foram detectadas %d falas de %d candidatos.", len(df), df["candidato"].nunique())

# === 3. Normalizar Texto ===
logging.info("Normalizando texto (lowercase, remoção de stopwords, etc.)...")
# Garante que stopwords estejam disponíveis
try:
    stopwords.words("portuguese")
except LookupError:
    nltk.download("stopwords", quiet=True)

stopwords_pt = set(stopwords.words("portuguese"))

def limpar(texto):
    texto = texto.lower()
    # Mantém letras latinas (inclui acentos) e espaços; remove números e pontuação
    texto = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ\s]", "", texto)
    palavras = [w for w in texto.split() if w not in stopwords_pt]
    return " ".join(palavras)

df["fala_limpa"] = df["fala"].apply(limpar)

# === 4. Detectar temas com TF-IDF ===
logging.info("Extraindo tópicos principais com TF-IDF...")
df_nao_vazio = df[df["fala_limpa"].str.strip() != ""].copy()
top_terms = []
if not df_nao_vazio.empty:
    try:
        tfidf = TfidfVectorizer(max_features=20)
        X = tfidf.fit_transform(df_nao_vazio["fala_limpa"])
        top_terms = tfidf.get_feature_names_out().tolist()
        logging.info("Principais termos detectados: %s", ", ".join(top_terms))

        freq = pd.DataFrame(X.toarray(), columns=top_terms)
        freq["candidato"] = df_nao_vazio["candidato"].values
        temas_por_candidato = freq.groupby("candidato").sum()
    except ValueError as ve:
        logging.warning("Falha ao extrair TF-IDF (vocabulário vazio): %s", ve)
        temas_por_candidato = pd.DataFrame()
else:
    logging.warning("Todos os documentos ficaram vazios após limpeza. Pulando etapa de TF-IDF.")
    temas_por_candidato = pd.DataFrame()

logging.info("Pipeline finalizada com sucesso ✅")

# === 8. Análise de TEMAS (dicionário) ===
logging.info("Iniciando análise de temas com base em dicionário...")

def remover_acentos(texto_base: str) -> str:
    return unicodedata.normalize("NFD", texto_base).encode("ascii", "ignore").decode("utf-8")

def normalizar_com_mapa(texto_base: str):
    """Normaliza removendo acentos (ASCII) e retorna o texto normalizado
    e um mapa de indices (posicao_normalizada -> posicao_original)."""
    norm_chars = []
    index_map = []
    for i, ch in enumerate(texto_base):
        decomposed = unicodedata.normalize("NFD", ch)
        # remove marcas de acento
        filtered = ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn')
        # mantém somente ASCII imprimível
        try:
            filtered_ascii = filtered.encode('ascii', 'ignore').decode('ascii')
        except Exception:
            filtered_ascii = ''
        if filtered_ascii:
            norm_chars.append(filtered_ascii)
            # mapeia cada caractere gerado para o índice original
            for _ in range(len(filtered_ascii)):
                index_map.append(i)
    return ''.join(norm_chars), index_map

texto_lower = texto.lower()
texto_norm, norm_to_orig = normalizar_com_mapa(texto_lower)

temas_padroes = {
    "educacao": {
        "frases": [
            r"ensino medio", r"ensino fundamental", r"educacao infantil", r"educacao basica",
            r"universidade", r"escola", r"professor", r"aluno", r"creche", r"merenda",
        ],
        "palavras": [
            r"educa\w*", r"escola\w*", r"ensino\w*", r"universidad\w*", r"professor\w*", 
            r"alun\w*", r"creche\w*", r"merend\w*", r"enem\b", r"prouni\b", r"fies\b",
            r"pronatec\w*", r"escola\w* tecnica", r"formacao\w*", r"qualificacao\w*",
        ],
    },
    "economia": {
        "frases": [r"reforma tributaria", r"plano real", r"inflacao", r"crescimento economico"],
        "palavras": [
            r"econom\w*", r"inflac\w*", r"empreg\w*", r"desempreg\w*", r"trabalh\w*", r"salari\w*",
            r"impost\w*", r"rend\w*", r"pib\b", r"crescimento\w*", r"juros\w*", r"cambio\w*",
            r"dolar\w*", r"real\b", r"dolar\b", r"mercado\w*", r"investiment\w*",
        ],
    },
    "armas_seguranca": {
        "frases": [r"seguranca publica", r"porte de arma\w*", r"posse de arma\w*", r"violencia"],
        "palavras": [
            r"arma\w*", r"desarmament\w*", r"polici\w*", r"homicid\w*", r"violenc\w*", r"seguranc\w*",
            r"crime\w*", r"criminal\w*", r"assassinat\w*", r"trafico\w*", r"drog\w*",
            r"policia\w* federal", r"policia\w* militar", r"forca\w* armada\w*",
        ],
    },
    "religiao": {
        "frases": [r"liberdade de culto", r"moral crist\w*", r"gracas a deus"],
        "palavras": [
            r"relig\w*", r"igreja\w*", r"deus\b", r"evangelic\w*", r"catolic\w*", r"pastor\w*", 
            r"biblia\w*", r"culto\w*", r"fe\b", r"oracao\b", r"padre", r"abenc\w", r"crist\b",
        ],
    },
    "moralidade_sexualidade": {
        "frases": [r"bons costumes", r"educacao sexual", r"familia tradicional", r"ideologia de genero"],
        "palavras": [
            r"moral\w*", r"famil\w*", r"valore\w*", r"costum\w*", r"tradic\w*", r"conservador\w*",
            r"sexual\w*", r"genero\w*", r"lgbt\w*", r"homossex\w*", r"transex\w*", r"aborto\w*", 
            r"sexo\b", r"drog\b", r"casamento\w*", r"divorcio\w*",
        ],
    },
    "democracia": {
        "frases": [r"liberdade de expressao", r"estado de direito", r"democracia"],
        "palavras": [
            r"democrac\w*", r"constituic\w*", r"stf\b", r"eleic\w*", r"urna\w*", r"golpe\w*", 
            r"ditadur\w*", r"instituic\w*", r"liberdade\w*", r"voto\w*", r"candidat\w*",
            r"partido\w*", r"politic\w*", r"governo\w*", r"presidente\w*",
        ],
    },
    "corrupcao": {
        "frases": [r"lava jato", r"cassacao do mandato", r"crime de responsabilidade", r"mensalao"],
        "palavras": [
            r"corrupc\w*", r"mensalao\b", r"rachadin\w*", r"propin\w*", r"desvi\w*", 
            r"lavagem de dinheiro", r"corrup\w*", r"improbidad\w*", r"peculato\b", 
            r"enriquecimento ilic\w*", r"laranj\w*", r"petrolao\b", r"delacao\w*",
        ],
    },
    "fake_news": {
        "frases": [
            r"fake news", r"noticia falsa", r"noticias falsas", r"checagem de fatos", 
            r"verificacao de fatos", r"fact checking",
        ],
        "palavras": [
            r"desinformacao\w*", r"misinformacao\w*", r"boato\w*", r"mentir\w*", r"mentira\w*", 
            r"engan\w*", r"fals\w*", r"propaganda\w*", r"calunia\w*", r"difamacao\w*",
        ],
    },
    "saude": {
        "frases": [
            r"saude publica", r"sistema unico de saude", r"atencao primaria a saude",
            r"passaporte vacinal", r"obrigatoriedade da vacina", r"unidade de terapia intensiva",
        ],
        "palavras": [
            r"saude\b", r"doenca\w*", r"pandemi\w*", r"covid\w*", r"coronavirus\w*", r"sars\w*cov\w*",
            r"vacina\w*", r"vacin\w*", r"imuniz\w*", r"dose\w*", r"sus\b", r"hospita\w*", 
            r"leito\w*", r"uti\b", r"oxigen\w*", r"respirad\w*", r"mascar\w*", r"quarenten\w*", 
            r"isolament\w*", r"lockdown\b", r"epidem\w*", r"contagi\w*", r"transmiss\w*",
            r"medic\w*", r"remedio\w*", r"tratament\w*",
        ],
    },
    "previdencia": {
        "frases": [r"previdencia social", r"aposentadoria", r"inss"],
        "palavras": [
            r"previdenc\w*", r"aposentador\w*", r"inss\b", r"aposentad\w*", r"pensionist\w*",
            r"fator previdenciario", r"reforma da previdencia", r"capitalizacao\w*",
        ],
    },
    "infraestrutura": {
        "frases": [r"obras publicas", r"infraestrutura", r"transporte publico"],
        "palavras": [
            r"infraestrutur\w*", r"obra\w*", r"construcao\w*", r"estrada\w*", r"rodovia\w*",
            r"ferrovia\w*", r"porto\w*", r"aeroporto\w*", r"energia\w*", r"agua\w*",
            r"saneament\w*", r"transporte\w*", r"metro\w*", r"onibu\w*",
        ],
    },
    "agricultura": {
        "frases": [r"agricultura familiar", r"agronegocio", r"reforma agraria"],
        "palavras": [
            r"agricultur\w*", r"agronegoci\w*", r"fazenda\w*", r"rural\w*", r"campo\w*",
            r"produtor\w* rural", r"agricultor\w*", r"reforma agraria", r"terra\w*",
            r"credito rural", r"financiamento\w* agricol\w*",
        ],
    },
    "meio_ambiente": {
        "frases": [r"meio ambiente", r"desenvolvimento sustentavel", r"mudanca climatica"],
        "palavras": [
            r"ambiente\w*", r"ecolog\w*", r"sustentavel\w*", r"clima\w*", r"poluicao\w*",
            r"floresta\w*", r"amazonia\w*", r"preservacao\w*", r"reciclagem\w*",
            r"energia renovavel", r"energia solar", r"energia eolica",
        ],
    },
    "internacional": {
        "frases": [r"relacoes internacionais", r"politica externa", r"mercosul"],
        "palavras": [
            r"internacional\w*", r"exterior\w*", r"mercosul\b", r"alca\b", r"brics\b",
            r"onu\b", r"fmi\b", r"banco mundial", r"exportacao\w*", r"importacao\w*",
            r"comercio exterior", r"diplomacia\w*",
        ],
    },
}

def contar_tema(texto_base: str, padroes: dict) -> int:
    restante = texto_base
    total = 0
    # Frases primeiro (evita dupla contagem)
    for fr in padroes.get("frases", []):
        matches = list(re.finditer(rf"\b{fr}\b", restante))
        total += len(matches)
        if matches:
            restante = re.sub(rf"\b{fr}\b", " ", restante)
    # Palavras
    for pw in padroes.get("palavras", []):
        total += len(re.findall(rf"\b{pw}\b", restante))
    return total

def extrair_ocorrencias(texto_norm: str, texto_orig: str, norm_to_orig: list, padroes: dict, janela: int = 120) -> list:
    """Retorna trechos do texto original que casam com os padrões do tema.
    janela: número de caracteres para expandir antes/depois da ocorrência no original.
    """
    trechos = []
    # compila padrões no texto normalizado
    for fr in padroes.get("frases", []):
        for m in re.finditer(rf"\b{fr}\b", texto_norm):
            a, b = m.start(), m.end()
            i0 = norm_to_orig[max(0, a)]
            i1 = norm_to_orig[min(len(norm_to_orig) - 1, b - 1)]
            s = max(0, i0 - janela)
            e = min(len(texto_orig), i1 + janela)
            trechos.append(texto_orig[s:e].strip())
    for pw in padroes.get("palavras", []):
        for m in re.finditer(rf"\b{pw}\b", texto_norm):
            a, b = m.start(), m.end()
            i0 = norm_to_orig[max(0, a)]
            i1 = norm_to_orig[min(len(norm_to_orig) - 1, b - 1)]
            s = max(0, i0 - janela)
            e = min(len(texto_orig), i1 + janela)
            trechos.append(texto_orig[s:e].strip())
    return trechos

temas_contagem = {tema: contar_tema(texto_norm, pats) for tema, pats in temas_padroes.items()}

temas_df = pd.DataFrame(
    sorted(temas_contagem.items(), key=lambda kv: kv[1], reverse=True),
    columns=["tema", "ocorrencias"],
)

palavras_totais = max(1, len(re.findall(r"\b\w+\b", texto_norm)))
temas_df["freq_por_10k"] = (temas_df["ocorrencias"] / palavras_totais) * 10000

logging.info(
    "Temas detectados (ocorrencias / por 10k palavras):\n%s",
    temas_df.to_string(index=False, formatters={"freq_por_10k": lambda x: f"{x:.2f}"}),
)

plt.figure(figsize=(10, 5))
sns.barplot(data=temas_df, x="tema", y="ocorrencias", palette="muted")
plt.title("Contagem de temas no debate (dicionário)")
plt.ylabel("Ocorrências")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
out_png_temas = os.path.join(os.path.dirname(__file__), "temas_contagem.png")
plt.savefig(out_png_temas, dpi=150)
plt.close()
logging.info("Gráfico de temas salvo em %s", out_png_temas)

# Salva contagem de temas em CSV para auditoria
out_csv_temas = os.path.join(os.path.dirname(__file__), "temas_contagem.csv")
temas_df.to_csv(out_csv_temas, index=False)
logging.info("Tabela de temas salva em %s", out_csv_temas)

# === Exporta extratos do texto original por tema ===
extratos = []
max_por_tema = 50  # limita para manter arquivo legível
for tema, padroes in temas_padroes.items():
    trechos = extrair_ocorrencias(texto_norm, texto_lower, norm_to_orig, padroes, janela=160)
    # remove duplicatas simples mantendo ordem
    vistos = set()
    trechos_unicos = []
    for t in trechos:
        k = t.strip()
        if k and k not in vistos:
            vistos.add(k)
            trechos_unicos.append(k)
        if len(trechos_unicos) >= max_por_tema:
            break
    for t in trechos_unicos:
        extratos.append({"tema": tema, "trecho": t})

if extratos:
    extratos_df = pd.DataFrame(extratos)
    out_csv_extratos = os.path.join(os.path.dirname(__file__), "temas_extratos.csv")
    extratos_df.to_csv(out_csv_extratos, index=False)
    logging.info("Extratos de temas salvos em %s (até %d por tema)", out_csv_extratos, max_por_tema)

# === 9. Descoberta de temas (LDA não supervisionado) ===
logging.info("Iniciando descoberta de temas com LDA (não supervisionado)...")

def split_text_into_chunks(text: str, words_per_chunk: int = 300) -> list:
    tokens = re.findall(r"\b\w+\b", text)
    chunks = []
    for i in range(0, len(tokens), words_per_chunk):
        chunk_tokens = tokens[i : i + words_per_chunk]
        if chunk_tokens:
            chunks.append(" ".join(chunk_tokens))
    return chunks

# Prepara documentos a partir de janelas do texto completo normalizado
docs = split_text_into_chunks(texto_norm, words_per_chunk=300)

if len(docs) < 2:
    logging.warning("Texto muito curto para LDA (docs=%d). Pulando etapa.", len(docs))
else:
    # Stopwords estendidas para remover termos pouco informativos do debate
    extra_stop = {
        # marcadores de debate e conectores
        "agora", "aqui", "entao", "falando", "debate", "bloco", "sera", "rodada", "minuto", "minutos",
        "responder", "jornalistas", "confronto", "perguntas", "pergunta", "replica", "tréplica", "tempo", 
        "primeiro", "segundo", "terceiro", "quarto", "quinto", "sexto", "sétimo", "oitavo", "nono", "décimo", "regras", "regra"
        # pronomes e palavras vazias frequentes
        "gente", "pessoas", "alguem", "nada", "coisa", "coisas", "nos", "nós", "voce", "vocês", "voces",
        "antes", "depois","nao", "ou", "ou seja", "se", "senao", "senao se", "senhor", "senhora",
        # verbos/ações muito gerais
        "sabe", "vou", "fez", "falar", "ter", "tive", "teve", "tem", "têm", "tendo", "vai", "vamos",
        "acho", "queria", "quero", "dizer", "disse", "falou", "falei", "falam", "tambem", "tbm",
        "fizemos", "fiz", "estar", "estamos", "estao", "foi", "foram", "foram", "questao", "houve", "forma"
        # política genérica
        "presidente", "candidato", "candidatos", "brasil", "pais", "porque", "sobre", "todos", "governo", "sao paulo", "minas gerais",
        "candidata", "candidatas",
        # nomes próprios (normalizados sem acento/maiusc)
        "bolsonaro", "lula", "jair", "jair bolsonaro", "marina", "marina silva", "cabo", "daciolo", "cabo daciolo"
        # nomes de redes sociais
        "twitter", "facebook", "instagram", "youtube", "tiktok", "x", "twitter brasil", "google"
        "facebook brasil", "instagram brasil", "youtube brasil", "tiktok brasil", "x brasil",
        
    }
    sw = set(stopwords_pt)
    sw.update(extra_stop)

    def preproc(text: str) -> str:
        # Já está normalizado; apenas remove stopwords curtas remanescentes
        return " ".join([t for t in text.split() if t not in sw and len(t) > 2])

    docs_clean = [preproc(d) for d in docs]

    # Vetorização com n-gramas para captar expressões
    vectorizer = CountVectorizer(
        max_df=0.90,
        min_df=5,
        max_features=10000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-z]{3,}\b",
    )

    try:
        dtm = vectorizer.fit_transform(docs_clean)
    except ValueError as ve:
        logging.warning("Falha ao vetorizar para LDA: %s", ve)
        dtm = None

    if dtm is not None and dtm.shape[1] > 0:
        # Seleção simples do número de temas pelo score do modelo
        candidate_k = [5, 7, 10, 12, 15]
        best_model = None
        best_score = float("-inf")
        for k in candidate_k:
            try:
                lda = LatentDirichletAllocation(
                    n_components=k,
                    learning_method="batch",
                    random_state=42,
                    n_jobs=-1,
                    evaluate_every=0,
                )
                lda.fit(dtm)
                score = lda.score(dtm)  # log-likelihood; maior é melhor
                if score > best_score:
                    best_score = score
                    best_model = lda
            except Exception as e:
                logging.debug("LDA k=%d falhou: %s", k, e)

        if best_model is None:
            logging.warning("Não foi possível ajustar LDA em nenhum k candidato.")
        else:
            lda = best_model
            feature_names = vectorizer.get_feature_names_out()

            def top_terms_for_topic(comp, topn=12):
                indices = comp.argsort()[::-1][:topn]
                return [feature_names[i] for i in indices]

            topics = []
            for topic_idx, comp in enumerate(lda.components_):
                terms = top_terms_for_topic(comp, topn=12)
                topics.append({"topic": topic_idx, "top_terms": ", ".join(terms)})

            topics_df = pd.DataFrame(topics)
            logging.info("Temas LDA (top termos):\n%s", topics_df.to_string(index=False))

            # Atribuição de tema dominante por chunk
            doc_topic = lda.transform(dtm)
            dominant_topic = doc_topic.argmax(axis=1)
            dist_df = pd.Series(dominant_topic).value_counts().sort_index()

            lda_topics_out = os.path.join(os.path.dirname(__file__), "lda_topics.csv")
            lda_assign_out = os.path.join(os.path.dirname(__file__), "lda_chunks_topics.csv")
            topics_df.to_csv(lda_topics_out, index=False)
            pd.DataFrame({"chunk": list(range(len(docs))), "topic": dominant_topic}).to_csv(lda_assign_out, index=False)
            logging.info("Tabelas LDA salvas em %s e %s", lda_topics_out, lda_assign_out)

            # Gráfico de distribuição dos temas
            plt.figure(figsize=(10, 5))
            sns.barplot(x=[f"T{int(i)}" for i in dist_df.index], y=dist_df.values, palette="crest")
            plt.title("Distribuição de temas (LDA)")
            plt.ylabel("Número de chunks dominantes")
            plt.xlabel("Tema")
            plt.tight_layout()
            out_png_lda = os.path.join(os.path.dirname(__file__), "lda_topics_distribution.png")
            plt.savefig(out_png_lda, dpi=150)
            plt.close()
            logging.info("Gráfico LDA salvo em %s", out_png_lda)
    else:
        logging.warning("Vocabulário insuficiente para LDA (matriz vazia).")
