# bio_utils.py

# --- Importaciones necesarias para las funciones ---
import pandas as pd
import subprocess
import os
import numpy as np
from pathlib import Path
from Bio import SeqIO
from typing import Optional, Dict, Any

IUPAC_AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWYBXZJUO-*")  # IUPAC amino acids + gap/stop

def save_df_as_fasta(dataframe: pd.DataFrame, id_col: str, seq_col: str, output_file: Path, verbose: bool = True):
    """
    Saves specified columns from a DataFrame into a FASTA formatted file.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the sequence data.
        id_col (str): The name of the column to be used for the FASTA header.
        seq_col (str): The name of the column containing the sequence.
        filename (str): The name of the output file.
    """
    try:
        with open(output_file, 'w') as fasta_file:
            # Iterate over each row in the DataFrame
            for index, row in dataframe.iterrows():
                identifier = row[id_col]
                sequence = row[seq_col]

                # Ensure the sequence is not null before writing
                if pd.notna(sequence) and sequence:
                    # Write the header and sequence to the file
                    fasta_file.write(f">{identifier}\n")
                    fasta_file.write(f"{sequence}\n")

        if verbose:
            print(f"Success! DataFrame has been saved to '{output_file}'.")

    except KeyError as e:
        print(f"Error: Column {e} not found. Please check your column names.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def run_clustal_omega(input_file: Path, output_file: Path, output_format: str = 'clu', threads: int = 4):
    """
    Runs a sequence alignment using the Clustal Omega command-line tool.

    Args:
        input_file (Path): The path to the input FASTA file.
        output_file (Path): The path where the output alignment file will be saved.
        output_format (str): The desired output format (e.g., 'clu', 'fasta', 'msf'). Defaults to 'clu'.
        threads (int): The number of CPU threads to use. Defaults to 4.

    Returns:
        bool: True if the alignment was successful, False otherwise.
    """
    # Determine the executable name based on the operating system
    clustalo_executable = "clustalo.exe" if os.name == 'nt' else "clustalo"

    try:
        print("\nRunning Clustal Omega... (This may take a while)")
        
        command = [
            clustalo_executable,
            "-i", str(input_file),
            "-o", str(output_file),
            f"--outfmt={output_format}", # Use the output_format parameter
            "--auto",
            "-v",
            "--force",
            f"--threads={threads}"
        ]
        
        # Run the command
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        print("\nAlignment complete!")
        print(f"Result saved to: '{output_file}'")
        return True

    except FileNotFoundError:
        print(f"ERROR: Executable '{clustalo_executable}' not found.")
        print("Please ensure Clustal Omega is installed and accessible in your system's PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Clustal Omega failed during execution.")
        print("Error message:\n", e.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def test():
    print("Test function in bio_utils.py is working!")
    
# This function requires Biopython and pandasimport pandas as pd
from Bio import SeqIO
from pathlib import Path
from typing import Optional, Dict, Any

def inspect_fasta_file(file_path: Path, iupac: bool = False, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """
    Inspects a FASTA file for format, critical errors, and warnings.

    Args:
        file_path (Path): The path to the file to be validated.
        verbose (bool): If True, prints detailed progress and results. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing inspection results:
                        {'is_valid': bool, 
                         'record_count': int,
                         'has_duplicates': bool,
                         'duplicate_ids': list,
                         'invalid_chars_found': dict,
                         'empty_sequences': list}
        None: If the file is not found or a critical parsing error occurs.
    """
    if verbose:
        print(f"Inspecting file: {file_path}...")

    # --- Initialization ---
    record_count = 0
    seen_ids = set()
    duplicate_ids = []
    invalid_chars_found = {}
    empty_sequences = []
    is_valid = True  # Tracks critical errors
    has_duplicates = False # Tracks warnings
    if iupac:
        valid_chars = IUPAC_AMINO_ACID_SET # IUPAC amino acids + gap/stop
    else:
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ-*")

    try:
        with open(file_path, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                record_count += 1
                
                # --- Validation Checks ---
                # 1. Critical Error: Empty sequence
                if len(record.seq) == 0:
                    empty_sequences.append(record.id)
                    is_valid = False

                # 2. Warning: Duplicate ID
                if record.id in seen_ids:
                    duplicate_ids.append(record.id)
                    has_duplicates = True
                seen_ids.add(record.id)

                # 3. Critical Error: Invalid characters
                sequence_chars = set(str(record.seq).upper())
                rogue_chars = sequence_chars.difference(valid_chars)
                if rogue_chars:
                    invalid_chars_found[record.id] = list(rogue_chars)
                    is_valid = False
            
            if record_count == 0:
                is_valid = False
                if verbose: print("  - ERROR: The file contains no FASTA records.")

    except FileNotFoundError:
        if verbose: print(f"  - ERROR: The file was not found at the specified path.")
        return None
    except Exception as e:
        if verbose: print(f"  - ERROR: Invalid FASTA format or parsing error: {e}")
        return None

    # --- Prepare and print results ---
    results = {
        'is_valid': is_valid,
        'record_count': record_count,
        'has_duplicates': has_duplicates,
        'duplicate_ids': list(set(duplicate_ids)),
        'invalid_chars_found': invalid_chars_found,
        'empty_sequences': empty_sequences
    }

    if verbose:
        if is_valid:
            print(f"  - OK! File is structurally valid. Found {record_count} records.")
        else:
            print(f"  - FAILED! File has critical errors.")
            if results['invalid_chars_found']:
                print(f"    - Invalid characters in records: {results['invalid_chars_found']}")
            if results['empty_sequences']:
                print(f"    - Records with empty sequences: {results['empty_sequences']}")
        
        if has_duplicates:
            print(f"  - WARNING: Found duplicate IDs: {results['duplicate_ids']}")

    return results


def fasta_to_dataframe(filename: str) -> pd.DataFrame:
    """
    Loads sequences from a FASTA file into a pandas DataFrame.

    Args:
        filename (str): The path to the FASTA file.

    Returns:
        pd.DataFrame: A DataFrame with 'id', 'description', and 'sequence' columns.
    """
    # SeqIO.parse is used to read the file.
    # 'fasta' indicates the file format.
    # This returns a generator, which is memory-efficient.
    fasta_sequences = SeqIO.parse(open(filename), 'fasta')
    
    # Create lists to store the data for each sequence
    ids = []
    descriptions = []
    sequences = []
    
    # Iterate over each record in the FASTA file
    for seq_record in fasta_sequences:
        ids.append(seq_record.id)
        descriptions.append(seq_record.description)
        sequences.append(str(seq_record.seq))
    
    # Create a dictionary from the lists
    data = {
        'id': ids,
        'description': descriptions,
        'sequence': sequences
    }
    
    # Convert the dictionary into a pandas DataFrame
    df = pd.DataFrame(data)
    
    return df

# --- Función para calcular la matriz de identidad ---
def calculate_identity_matrix(sequences):
    """
    Calcula una matriz de identidad por pares para una lista de secuencias de igual longitud.
    """
    n_sequences = len(sequences)
    identity_matrix = np.zeros((n_sequences, n_sequences))
    
    for i in range(n_sequences):
        for j in range(i, n_sequences):
            # Calculte la identidad entre la secuencia i y j
            seq1 = sequences[i]
            seq2 = sequences[j]
            if len(seq1) == 0: continue # Evitar división por cero
            
            score = sum(1 for a, b in zip(seq1, seq2) if a == b)
            identity = score / len(seq1)
            
            # La matriz es simétrica
            identity_matrix[i, j] = identity
            identity_matrix[j, i] = identity
            
    return identity_matrix