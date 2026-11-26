import numpy as np
from pathlib import Path

def init_metrics():
    metrics = {
        '1': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        '-1': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'macro avg': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'weighted avg': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'accuracy': [],
        'time': []
    }
    return metrics


def save_values(metrics, values):
    for key in metrics.keys():
      if key == 'accuracy' or key == 'time':
        metrics[key].append(values[key])
      else:
        for key2 in metrics[key].keys():
          metrics[key][key2].append(values[key][key2])


def write_results(metrics, file_name, line_parameters, path):
    # 1. Âncora o caminho base onde o próprio utils.py está guardado
    diretorio_atual = Path(__file__).resolve().parent
    
    # 2. Constrói o caminho para a pasta 'results' (OLGA/codes/results)
    pasta_resultados = diretorio_atual / '..' / 'results'
    
    # 3. Cria a pasta automaticamente se ela não existir
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    
    # 4. Junta a pasta com o nome do ficheiro de forma segura
    caminho_final = pasta_resultados / file_name

    # Substituímos o (path + file_name) por caminho_final
    if not caminho_final.is_file():
        file_ = open(caminho_final, 'w')
        string = 'Parameters'
        for key in metrics.keys():
            if key == 'accuracy' or key == 'time':
              string += ';' + key + '-mean;' + key + '-std'
            else:
              for key2 in metrics[key].keys():
                string += ';' + key + '_' + key2 + '-mean;' + key + '_' + key2 + '-std'

        string += '\n'
        file_.write(string)
        file_.close()

    file_ = open(caminho_final, 'a')
    string = line_parameters

    for key in metrics.keys():
      if key == 'accuracy' or key == 'time':
        string += ';' + str(np.mean(metrics[key])) + ';' + str(np.std(metrics[key]))
      else:
        for key2 in metrics[key].keys():
          string += ';' + str(np.mean(metrics[key][key2])) + ';' + str(np.std(metrics[key][key2]))

    string += '\n'
    file_.write(string)
    file_.close()
