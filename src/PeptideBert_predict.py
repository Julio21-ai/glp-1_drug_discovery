import sys
import os
import yaml
import torch 
import pandas as pd
from tqdm.auto import tqdm
from models.peptideBert.network import create_model


def load_bert_model(model_directory_path, feature, device):
    config = yaml.load(open(f'{model_directory_path}/{feature}/config.yaml', 'r'), Loader=yaml.FullLoader)
    config['device'] = device
    model = create_model(config)
    model.load_state_dict(torch.load(f'{model_directory_path}/{feature}/model.pt',weights_only = False)['model_state_dict'], strict=False)
    model.to(device)
    return model



def predict_peptidebert(model_directory_path, 
                        input_dataframe: pd.DataFrame, 
                        sequence_col: str = 'sequence', 
                        feats=['hemo','sol','nf']):
    """
    Ejecuta PeptideBERT sobre un DataFrame y añade las predicciones como nuevas columnas.

    Args:
        model_directory_path (str): Ruta al directorio de modelos.
        input_dataframe (pd.DataFrame): DataFrame que contiene las secuencias.
        sequence_col (str, optional): Nombre de la columna que tiene las secuencias. 
                                      Defaults to 'sequence'.
        feats (list, optional): Lista de features a predecir. 
                                Defaults to ['hemo','sol','nf'].

    Returns:
        pd.DataFrame: Una copia del DataFrame original con las columnas de predicción añadidas.
    """
    
    # 1. Extraer la lista de secuencias del DataFrame
    sequences_list = input_dataframe[sequence_col].tolist()
    
    # Manejar caso de DataFrame vacío
    if not sequences_list:
        print("DataFrame de entrada está vacío. Devolviendo copia.")
        results = input_dataframe.copy()
        for c in feats:
            results[c] = pd.Series(dtype=float) # Añade columnas vacías
        return results

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')
    
    MAX_LEN = max(map(len, sequences_list))
    
    # Definición del mapping
    mapping = dict(zip(
        ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','L',
         'A','G','V','E','S','I','K','R','D','T','P','N',
         'Q','F','Y','M','H','C','W'],
        range(30)
    ))

    # 2. Tokenizar las secuencias (usamos una copia de la lista)
    tokenized_sequences = []
    for seq in sequences_list:
        tokenized_seq = [mapping.get(c, mapping['[UNK]']) for c in seq] # Usar .get para manejar UNK
        tokenized_seq.extend([0] * (MAX_LEN - len(tokenized_seq)))  # padding
        tokenized_sequences.append(tokenized_seq)
    
    # 3. Crear el DataFrame de resultados a partir de una COPIA del original
    #    Esto conserva todas las columnas (como 'ID') y el índice original.
    results = input_dataframe.copy()
    
    with torch.inference_mode():

        for c in feats:
            try:
                model = load_bert_model(model_directory_path, c, device)
                preds = []
                #for i in tqdm(range(len(prompt_sequences))):
                # Iterar sobre la lista de secuencias tokenizadas
                print(f"Procesando caracteristica: {c}")
                for i in tqdm(range(len(tokenized_sequences))):
                    input_ids = torch.tensor([tokenized_sequences[i]]).to(device)
                    attention_mask = (input_ids != 0).float()
                    output = float(model(input_ids, attention_mask)[0])
                    preds.append(output)
                                   
                # 4. Añadir la lista de predicciones como una NUEVA COLUMNA
                #    Usar pd.Series con el índice de 'results' asegura que todo
                #    se alinee correctamente, incluso si el DataFrame tiene un índice personalizado.
                results[c] = pd.Series(preds, index=results.index, dtype=float)
            except Exception as e:
                print(f"Error durante la predicción con PeptideBERT: {e}")
                raise e
            finally:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                if model is not None:
                    del model                

    return results