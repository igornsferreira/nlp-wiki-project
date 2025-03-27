from nlp_utils import escolher_idioma, get_verbete, processar_texto, nuvem_palavras, resumo_textual, buscar

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
