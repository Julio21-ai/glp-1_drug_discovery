# procesador_ifeature.py

# Importaciones necesarias para que las funciones operen de forma independiente.
from iFeatureOmega import iFeatureOmegaCLI
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from typing import List, Optional, Dict, Any
from .bio_utils import save_df_as_fasta, fasta_to_dataframe, inspect_fasta_file

def compute_single_descriptor(input_fasta_file, descriptor, settings_json_file=None):
    """
    Calcula un descriptor con iFeatureOmega y devuelve un DataFrame indexado por ID.

    Parámetros
    ----------
    input_fasta_file : str
        Ruta al archivo FASTA o TXT con secuencias.
    descriptor : str
        Nombre del descriptor (por ejemplo, "AAC", "DPC", "CTDC").
    settings_json_file : str | None
        Ruta al archivo JSON de configuración de parámetros.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con las secuencias como índice y columnas prefijadas con el descriptor.
    """
    #print(f"Calculando descriptor: {descriptor}")

    protein = iFeatureOmegaCLI.iProtein(input_fasta_file)

    if settings_json_file:
        try:
            protein.import_parameters(settings_json_file)
        except Exception as e:
            print(f"No se pudo importar parámetros: {e}")

    try:
        protein.get_descriptor(descriptor)
        df = protein.encodings.reset_index()
    except Exception as e:
        print(f"Error al calcular {descriptor}: {protein.encodings} {e}")
        return None



    if df.empty:
        print(f"Descriptor {descriptor} no generó resultados.")
        return None

    # Normalizar el nombre de la columna ID y ponerla como índice
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)
    df = df.set_index("ID")
    return df

def compute_peptide_features(input_fasta_file, descriptors, settings_json_file, output_csv=None):
    """
    Calcula una lista de descriptores con iFeatureOmega y devuelve un DataFrame combinado.
    """
    if not descriptors or not isinstance(descriptors, (list, tuple)):
        raise ValueError("Se necesita una lista no vacía de descriptores.")

    results = []
    for i in tqdm(range(0, len(descriptors)), desc="calculando descriptores"):
        desc = descriptors[i]
        df = compute_single_descriptor(input_fasta_file, desc, settings_json_file)
        if df is not None:
            results.append((desc, df))

    if not results:
        raise Exception("No se pudieron calcular descriptores válidos.")

    # Combina todos los dataframes de resultados en uno solo
    combined_df = results[0][1]

    for desc, df in results[1:]:
        combined_df = combined_df.merge(df, left_index=True, right_index=True, how="inner")

    combined_df.reset_index(inplace=True)

    if output_csv:
        combined_df.to_csv(output_csv, index=False)
        print(f"Resultados guardados en {output_csv}")

    # # Opcional: Verificar si se perdieron secuencias durante el merge
    # expected_ids = results[0][1].index
    # for desc, df in results[1:]:
    #     lost = set(expected_ids) - set(df.index)
    #     if lost:
    #         print(f"En el descriptor {desc} faltaron {len(lost)} secuencias: {list(lost)[:3]}...")

    return combined_df

def calcular_descriptores_ifeature(
    directorio_temporal: Path,
    dataframe: pd.DataFrame,
    sequence_col: str,
    id_col: str,
    descriptores: Optional[List[str]] = None,
    ifeatures_settings_json: Optional[Path] = None
) -> pd.DataFrame:
    """
    Calcula descriptores de péptidos usando iFeature para un DataFrame dado.

    La función realiza los siguientes pasos:
    1. Genera un archivo FASTA temporal a partir del DataFrame.
    2. Valida el archivo FASTA.
    3. Define una lista de descriptores por defecto si no se proporciona una.
    4. Valida la existencia del archivo de configuración de iFeature (si se proporciona).
    5. Ejecuta 'compute_peptide_features' (wrapper de iFeature) sobre el FASTA.
    6. Une los descriptores calculados de nuevo al DataFrame original.
    7. Limpia el archivo FASTA temporal.

    Args:
        directorio_temporal: Objeto Path al directorio donde se guardarán los
                             archivos FASTA temporales.
        dataframe: DataFrame de Pandas que contiene los datos de las secuencias.
        sequence_col: Nombre de la columna en 'dataframe' que contiene las
                      secuencias de péptidos.
        id_col: Nombre de la columna en 'dataframe' que contiene los
                identificadores únicos para el FASTA.
        descriptores: Lista opcional de nombres de descriptores a calcular
                      por iFeature. Si es None, se usa una lista predefinida.
        ifeatures_settings_json: Ruta opcional al archivo JSON de configuración
                                 de iFeature.

    Returns:
        Un nuevo DataFrame que contiene los datos del 'dataframe' original
        unidos con los nuevos descriptores calculados.

    Raises:
        ValueError: Si el archivo FASTA temporal generado no es válido.
        FileNotFoundError: Si se proporciona 'ifeatures_settings_json' pero
                           el archivo no existe.
    """
    
    # 1. Definir la lista de descriptores por defecto si no se proporciona
    if descriptores is None:
        print("Usando lista de descriptores por defecto.")
        descriptores = [
                    "AAC",				# Amino acid composition
                    "CKSAAGP type 1",	# Composition of k-spaced amino acid group pairs type 1- normalized
                    "DPC type 1",		# Dipeptide composition type 1 - normalized
                    "CTDC",				# Composition
                    "CTDT",				# Transition
                    "CTDD",				# Distribution
                    "CTriad",			# Conjoint triad
                    "GAAC",				# Grouped amino acid composition
                    "Moran",			# Moran
                    "SOCNumber",		# Sequence-order-coupling number
                    "QSOrder",			# Quasi-sequence-order descriptors
                    "PAAC",				# Pseudo-amino acid composition
                    "APAAC",			# Amphiphilic PAAC
                    "NMBroto",			# Auto-cross covariance
                ]

    # 2. Validar el archivo de configuración si se proporciona
    if ifeatures_settings_json and not ifeatures_settings_json.exists():
        raise FileNotFoundError(
            f"El archivo de configuración de iFeature no se encontró en: {ifeatures_settings_json}"
        )

    # 3. Generar un nombre de archivo FASTA temporal y único
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    nombre_archivo_fasta = f"ifeature_input_{timestamp}.fasta"
    ruta_salida_fasta = directorio_temporal / nombre_archivo_fasta

    try:
        # 4. Guardar el DataFrame como archivo FASTA
        save_df_as_fasta(
            dataframe=dataframe,
            id_col=id_col,
            seq_col=sequence_col,
            output_file=ruta_salida_fasta
        )

        # 5. Validar el archivo FASTA creado
        results = inspect_fasta_file(ruta_salida_fasta)
        if not (results and results.get('is_valid')):
            print(f"\nLa validación falló para '{ruta_salida_fasta}'. Abortando.")
            print(f"Detalles: {results.get('errors', 'Error desconocido')}")
            raise ValueError(f"El archivo FASTA generado '{ruta_salida_fasta}' no es válido.")
        
        print(f"'{ruta_salida_fasta}' es válido. Se encontraron {results['record_count']} registros.")

        # 6. Calcular los descriptores usando iFeature
        print(f"Iniciando cálculo de {len(descriptores)} descriptores...")
        df_descriptores_ifeature = compute_peptide_features(
            ruta_salida_fasta,
            descriptores,
            ifeatures_settings_json
        )
        print("Cálculo de descriptores finalizado.")

        # 7. Unir los descriptores al DataFrame original
        # iFeature siempre devuelve una columna 'ID' que coincide con los headers del FASTA
        # Hacemos el 'merge' usando la columna 'id_col' original y la columna 'ID' de iFeature
        df_final = pd.merge(
            left=df_descriptores_ifeature,
            right=dataframe,
            left_on='ID',     # Clave del DataFrame de iFeature
            right_on=id_col,  # Clave del DataFrame original
            how='inner'
        )

        # Si el id_col original no era 'ID', tendremos dos columnas de ID.
        # Eliminamos la columna 'ID' de iFeature para evitar redundancia.
        if id_col != 'ID' and 'ID' in df_final.columns:
            df_final = df_final.drop(columns='ID')
            
        print(f"Descriptores unidos al DataFrame. Forma final: {df_final.shape}")

        df_final.columns = df_final.columns.str.replace('.', '_', regex=False)
        
        return df_final

    finally:
        # 8. Limpieza: Asegurarse de borrar el archivo temporal
        if ruta_salida_fasta.exists():
            try:
                ruta_salida_fasta.unlink()
                print(f"Archivo temporal '{ruta_salida_fasta}' eliminado.")
            except OSError as e:
                print(f"Error al eliminar el archivo temporal '{ruta_salida_fasta}': {e}")