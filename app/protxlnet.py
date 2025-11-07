import torch
import re  # Importamos 're' para la limpieza
from transformers import XLNetTokenizer, XLNetLMHeadModel
import argparse
import time
import sys
from pathlib import Path
notebook_dir = Path.cwd()
directorio_base = Path.cwd().parent
sys.path.append(str(directorio_base))

# --- Constantes de Aminoácidos para la limpieza ---
AMINO_ACIDS_STRING = "ACDEFGHIKLMNPQRSTVWY"
# El patrón de regex para eliminar cualquier cosa que NO sea un AA
pattern_to_remove = f"[^{AMINO_ACIDS_STRING}]"

def main():
    """
    Script interactivo para generar secuencias de péptidos
    usando un modelo ProtXLNet (fine-tuned) en un bucle.
    """
    
    # --- 1. Definir Argumentos de la Sesión ---
    parser = argparse.ArgumentParser(
        description="Sesión interactiva para generar secuencias con ProtXLNet."
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=Path(directorio_base/"models"/"prot_xlnet_finetuned"), # Asume un nombre de carpeta
        help="Ruta al directorio del modelo ProtXLNet fine-tuned."
    )
    parser.add_argument(
        "-n", "--num_sequences",
        type=int,
        default=5,
        help="Número de secuencias a generar POR CADA prompt."
    )
    parser.add_argument(
        "-l", "--max_length",
        type=int,
        default=100,
        help="Longitud máxima total de la secuencia generada."
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Temperatura para controlar la 'creatividad' (sampling)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=950, # ProtXLNet suele usar valores altos de Top-K
        help="Top-K sampling."
    )
    parser.add_argument(
        "--rep_penalty",
        type=float,
        default=1.0, # ProtXLNet es menos propenso a repetir
        help="Penalización por repetición."
    )
    
    args = parser.parse_args()

    # --- 2. Cargar Modelo y Tokenizer (UNA SOLA VEZ) ---
    print(f"Cargando modelo ProtXLNet desde {args.model_path}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = XLNetLMHeadModel.from_pretrained(Path(args.model_path)).to(device)
        tokenizer = XLNetTokenizer.from_pretrained(Path(args.model_path))
        
        # --- Configuración Específica de ProtXLNet para Generación ---
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token
        
    except Exception as e:
        print(f"¡Error! No se pudo cargar el modelo desde '{args.model_path}'.")
        print(f"Detalle: {e}")

        return
    
    try:
        model.eval()
        print(f"Modelo cargado en: {device}.")
        print("\n" + "="*50)
        print("   Sesión Interactiva de Generación ProtXLNet")
        print(f"   Modelo: {args.model_path}")
        print(f"   Generando {args.num_sequences} secuencias por prompt.")
        print("   Escribe 'salir' o 'exit' para terminar.")
        print("="*50 + "\n")

        # --- 3. Bucle Interactivo de Generación ---

        while True:
            # 3.1. Pedir el prompt al usuario
            prompt_text = input("Escribe tu prompt (dejar vacío para 'de novo'): ")
            prompt_text = prompt_text.strip().upper()

            if prompt_text.lower() in ['salir', 'exit', 'quit']:
                print("Terminando sesión...")
                break
                
            # --- 3.2. Preparar el Prompt (Formato ProtXLNet) ---
            if not prompt_text:
                formatted_prompt = "" # Generación 'de novo'
                print("Generando 'de novo'...")
            else:
                # Convertir "MGLSD" a "M G L S D"
                formatted_prompt = " ".join(list(prompt_text))
                print(f"Generando desde: '{formatted_prompt}'...")
                
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

            # 3.3. Generar Secuencias
            t_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],    
                    min_length=10,  # Longitud mínima razonable
                    max_length=args.max_length,
                    do_sample=True,
                    temperature=args.temp,
                    top_k=args.top_k,
                    repetition_penalty=args.rep_penalty,
                    num_return_sequences=args.num_sequences,
                    pad_token_id=tokenizer.pad_token_id # Usar el pad_token real
                )
            t_end = time.time()
            print(f"\nGenerado en {t_end - t_start:.2f}s:")

            # 3.4. Decodificar y Limpiar Salida
            # skip_special_tokens=True quita <s>, </s>, <pad>
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True) 
            
            for i, seq in enumerate(decoded):
                # 1. Quitar espacios: "M G L S D K" -> "MGLSDK"
                no_spaces = seq.replace(" ", "").replace("\n", "").strip()
                # 2. Quitar cualquier otro carácter no-AA (por si acaso)
                clean = re.sub(pattern_to_remove, '', no_spaces)
                print(f"[{i+1}] {clean}")
            
            print("\n" + "-"*50 + "\n") # Separador
    except KeyboardInterrupt:
        print("\nSesión interrumpida por el usuario.")
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer

    print("Sesión terminada. ¡Adiós!")

if __name__ == "__main__":
    main()