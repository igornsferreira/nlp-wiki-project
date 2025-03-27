import spacy
from spacy.matcher import Matcher, PhraseMatcher
import nltk
from goose3 import Goose
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

nltk.download('punkt_tab')

pt_spacy = spacy.load('pt_core_news_lg')
en_spacy = spacy.load('en_core_web_lg')

os.makedirs("pdfs", exist_ok=True)

def escolher_idioma():
    """Permite ao usuário escolher o idioma do processamento."""
    while True:
        idioma = input('Escolha o idioma (pt/en): ').strip().lower()
        if idioma == 'pt':
            return 'pt', pt_spacy 
        elif idioma == 'en':
            return 'en', en_spacy  
        else:
            print('Idioma Inválido. Digite "pt" para português ou "en" para inglês.')

def get_verbete(entrada, idioma):
    """Busca um verbete da Wikipedia ou extrai conteúdo de uma URL"""
    if entrada.startswith('http'):
        g = Goose()  
        article = g.extract(entrada)  
        return article.cleaned_text, article.title  
    else:
        url = f'https://{idioma}.wikipedia.org/wiki/{entrada}'  
        g = Goose()  
        article = g.extract(url)  
        return article.cleaned_text, article.title 

def processar_texto(texto, idioma):
    """Remove stopwords e pontuações do texto extraído"""
    nlp = pt_spacy if idioma == 'pt' else en_spacy
    doc = nlp(texto)
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def salvar_pdf(nome_arquivo):
    """Salva a última figura do Matplotlib como PDF"""
    caminho = f"pdfs/{nome_arquivo}"
    plt.savefig(caminho, format="pdf")
    print(f"PDF salvo em: {caminho}")

def nuvem_palavras(texto):
    """Gera e salva uma nuvem de palavras como PDF"""
    cloud = WordCloud(width=800, height=800, background_color='black').generate(texto)
    plt.figure(figsize=(8,8))
    plt.imshow(cloud, interpolation='bilinear')  
    plt.title("Nuvem de Palavras", fontsize=15, fontweight="bold", pad=15, color="red")
    plt.axis('off')  
    salvar_pdf("nuvem_palavras.pdf")
    plt.show()

def resumo_textual(texto, idioma):
    """Gera um resumo do texto original."""
    quantidade = int(input("Informe a quantidade de frases que deseja exibir no resumo: \n"))

    if idioma == "pt":
        parser = PlaintextParser.from_string(texto, Tokenizer("portuguese"))
    else:
        parser = PlaintextParser.from_string(texto, Tokenizer("english"))

    summarizer = SumBasicSummarizer()
    sumary = summarizer(parser.document, quantidade)

    resumo_texto = "\n".join([str(sentence) for sentence in sumary])
    
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, resumo_texto, ha="center", va="center", wrap=True, fontsize=12)
    plt.axis("off")
    plt.title(f"Resumo Textual", fontsize=20, fontweight="bold", pad=20, color="red")
    plt.tight_layout()
    salvar_pdf("resumo_textual.pdf")
    plt.show()

    return resumo_texto

def buscar(texto, idioma):
    """Busca um termo no texto original e exibe o contexto."""
    termo = input("Digite o termo que deseja buscar no texto: ").strip().lower()

    nlp = pt_spacy if idioma == "pt" else en_spacy
    doc = nlp(texto) 

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("TERMO_BUSCA", [nlp.make_doc(termo)])

    matches = matcher(doc)
    resultados = ""

    if matches:
        print(f"\n🔍 Termo '{termo}' encontrado em:\n")
        number_of_words = 5  
        for match_id, start, end in matches:
            inicio = max(0, start - number_of_words)
            fim = min(len(doc), end + number_of_words)
            contexto_texto = " ".join([token.text for token in doc[inicio:fim]])
            resultados += f"... {contexto_texto} ...\n"
            print(f"... {contexto_texto} ...\n")

        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, resultados, fontsize=12, ha="center", va="center", wrap=True)
        plt.axis("off")
        plt.title("Resultado da Busca", fontsize=20, fontweight="bold", pad=20, color="red")
        plt.tight_layout()
        salvar_pdf("busca_termo.pdf")
        plt.show()
    else:
        print(f"\nO termo '{termo}' não foi encontrado no texto.")
        resultados = f"O termo '{termo}' não foi encontrado no texto."

def exibir_menu():
    """Exibe o menu formatado"""
    print("\n=== MENU ===")
    print("[1] - Apresentar nuvem de palavras 📊")
    print("[2] - Exibir resumo textual 📝")
    print("[3] - Pesquisar por termo 🔎")
    print("[4] - Sair 🚪")

def main():
    """Executa o programa principal"""
    print("\n=== Processador de Texto com NLP ===")
    idioma, _ = escolher_idioma()
    entrada = input("\nDigite o verbete da Wikipedia ou a URL da página de interesse: ")
    texto, title = get_verbete(entrada, idioma)
    texto_processado = processar_texto(texto, idioma)
    print(f"\n✅ Tema escolhido: {title}")

    while True:
        exibir_menu()
        opcao = input("\nEscolha uma opção: ")
        
        if opcao == "1":
            nuvem_palavras(texto_processado)
        elif opcao == "2":
            resumo_textual(texto, idioma)
        elif opcao == "3":
            buscar(texto, idioma)
        elif opcao == "4":
            print("✅ Programa finalizado. Até logo!")
            break
        else:
            print("❌ Opção inválida! Tente novamente.")

if __name__ == "__main__":
    main()