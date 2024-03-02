import subprocess

def install_packages_from_file(file_path):
    with open(file_path, 'r') as file:
        for package in file:
            package = package.strip()
            if package:
                try:
                    subprocess.check_call(['pip', 'install', package])
                    print(f"O pacote '{package}' foi instalado com sucesso.")
                except subprocess.CalledProcessError:
                    print(f"Falha ao instalar o pacote '{package}'. Verifique o nome do pacote e a conexão com a internet.")

if __name__ == "__main__":
    # Substitua 'caminho/do/arquivo.txt' pelo caminho real do seu arquivo de texto.
    file_path = "C:\\Users\\LEONARDO\\Desktop\\Codigos\\Pacotes Python.txt"
    install_packages_from_file(file_path)