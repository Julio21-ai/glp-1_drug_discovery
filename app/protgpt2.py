import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time
import sys
from pathlib import Path
notebook_dir = Path.cwd()
directorio_base = Path.cwd().parent
sys.path.append(str(directorio_base))


def main():
    """
    Script interactivo para generar secuencias de péptidos
    usando un modelo ProtGPT2 (fine-tuned) en un bucle.
    """
    
    # --- 1. Definir Argumentos de la Sesión ---
    # (Estos se configuran UNA VEZ al iniciar el script)
    parser = argparse.ArgumentParser(
        description="Sesión interactiva para generar secuencias con ProtGPT2."
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=Path(directorio_base/"models"/"protgpt2_finetuned"),
        help="Ruta al directorio del modelo fine-tuned."
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
        default=950,
        help="Top-K sampling."
    )
    parser.add_argument(
        "--rep_penalty",
        type=float,
        default=1.2,
        help="Penalización por repetición."
    )
    
    args = parser.parse_args()

    # --- 2. Cargar Modelo y Tokenizer (UNA SOLA VEZ) ---
    print(f"Cargando modelo desde {args.model_path}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = AutoModelForCausalLM.from_pretrained(Path(args.model_path)).to(device)
        tokenizer = AutoTokenizer.from_pretrained(Path(args.model_path))
        
        tokenizer.eos_token_id = 0
        tokenizer.pad_token = tokenizer.eos_token
        
    except Exception as e:
        print(f"¡Error! No se pudo cargar el modelo desde '{args.model_path}'.")
        print(f"Detalle: {e}")
        return
    
    try:

        model.eval()
        print(f"Modelo cargado en: {device}.")
        total_params = model.num_parameters()
        print("\n" + "="*50)
        print("   Sesión Interactiva de Generación ProtGPT2")
        print(f"   Modelo: {args.model_path}")
        print(f"   Parámetros: {total_params:,}")
        print(f"   Generando {args.num_sequences} secuencias por prompt.")
        print("   Escribe 'salir' o 'exit' para terminar.")
        print("="*50 + "\n")

        # --- 3. Bucle Interactivo de Generación ---
        while True:
            # 3.1. Pedir el prompt al usuario
            prompt_text = input("Escribe tu prompt (dejar vacío para 'de novo'): ")
            prompt_text = prompt_text.strip()

            # 3.2. Condición de salida
            if prompt_text.lower() in ['salir', 'exit', 'quit']:
                print("Terminando sesión...")
                break
                
            # 3.3. Preparar el prompt
            formatted_prompt = f"<|endoftext|>\n{prompt_text.upper()}"
            if not prompt_text:
                print("Generando 'de novo'...")
            else:
                print(f"Generando desde: '{prompt_text.upper()}'...")
                
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

            # 3.4. Generar Secuencias
            t_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=args.max_length,
                    do_sample=True,
                    temperature=args.temp,
                    top_k=args.top_k,
                    repetition_penalty=args.rep_penalty,
                    num_return_sequences=args.num_sequences,
                    pad_token_id=tokenizer.eos_token_id
                )
            t_end = time.time()
            print(f"\nGenerado en {t_end - t_start:.2f}s:")

            # 3.5. Decodificar y Mostrar
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i, seq in enumerate(decoded):
                clean = seq.replace(" ", "").replace("\n", "").strip()
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