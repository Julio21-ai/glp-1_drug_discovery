# procesador_ifeature.py

# Importaciones necesarias para que las funciones operen de forma independiente.
from iFeatureOmega import iFeatureOmegaCLI
import pandas as pd

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
    print(f"Calculando descriptor: {descriptor}")

    protein = iFeatureOmegaCLI.iProtein(input_fasta_file)

    if settings_json_file:
        try:
            protein.import_parameters(settings_json_file)
        except Exception as e:
            print(f"No se pudo importar parámetros: {e}")

    try:
        protein.get_descriptor(descriptor)
    except Exception as e:
        print(f"Error al calcular {descriptor}: {e}")
        return None

    df = protein.encodings.reset_index()

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
    for desc in descriptors:
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