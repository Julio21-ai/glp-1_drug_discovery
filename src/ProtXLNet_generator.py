# peptide_generator.py

import random
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

# Lista de aminoácidos estándar para realizar las mutaciones.
# Se usa para asegurar que la mutación introducida sea válida.
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def generate_peptide_variants(
    prompt_sequences: list,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_variants_per_seq: int = 5,
    min_length: int = 25,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 3,
    top_p: float = 0.95
) -> list:
    """
    Genera nuevas variantes de péptidos introduciendo mutaciones aleatorias en
    un conjunto de secuencias de entrada y usando el modelo para completarlas.

    Args:
        prompt_sequences (list): Una lista de secuencias de péptidos (sin espacios)
                                 para usar como base para las mutaciones.
        model (PreTrainedModel): El modelo de lenguaje afinado (fine-tuned).
        tokenizer (PreTrainedTokenizer): El tokenizador correspondiente al modelo.
        num_variants_per_seq (int): Cuántas variantes generar por cada secuencia de entrada.
        min_length (int): La longitud mínima de las secuencias generadas (incluyendo el prompt).
        max_length (int): La longitud máxima de las secuencias generadas (incluyendo el prompt).
        temperature (float): Controla la aleatoriedad de la generación.
        top_k (int): Filtra los k tokens más probables.
        top_p (float): Filtra los tokens con probabilidad acumulada p (nucleus sampling).

    Returns:
        list: Una lista de secuencias de péptidos únicas generadas (sin espacios).
    """
    model.eval()
    device = model.device
    unique_variants = set()

    print(f"Iniciando generación de variantes para {len(prompt_sequences)} secuencias base...")

    for base_seq in tqdm(prompt_sequences, desc="Procesando secuencias"):
        if not base_seq or len(base_seq) < 1:
            continue

        for _ in range(num_variants_per_seq):
            # 1. Crear una mutación aleatoria
            seq_list = list(base_seq)
            mutation_index = random.randint(0, len(seq_list) - 1)
            original_aa = seq_list[mutation_index]
            
            new_aa = random.choice([aa for aa in AMINO_ACIDS if aa != original_aa])
            seq_list[mutation_index] = new_aa
            mutated_seq = "".join(seq_list)

            # 2. Preparar el prompt para el modelo (con espacios)
            prompt = " ".join(list(mutated_seq))
            print(f"\nPrompt generado: {prompt}")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            print(f"\nshape len: {inputs['input_ids'].shape[1]}")

            # 3. Generar variantes usando el modelo
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    min_length=min_length,
                    max_length=max_length,                    
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            # 4. Decodificar, limpiar y guardar la nueva variante
            for output in outputs:
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                cleaned_variant = generated_text.replace(" ", "")
                unique_variants.add(cleaned_variant)

    print(f"\nGeneración completada. Se encontraron {len(unique_variants)} variantes únicas.")
    return list(unique_variants)



def generate_peptide_variants_fast(
    prompt_sequences: list,
    model,
    tokenizer,
    num_variants_per_seq: int = 5,
    num_return_sequences: int=5,
    min_length: int = 25,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 3,
    top_p: float = 0.95,
    repetition_penalty: float=1.0,
    batch_size: int = 32
) -> list:
    model.eval()
    device = model.device
    unique_variants = set()

    # --- Pre-generar todas las mutaciones ---
    mutated_prompts = []
    for base_seq in prompt_sequences:
        if not base_seq:
            continue
        for _ in range(num_variants_per_seq):
            seq_list = list(base_seq)
            idx = random.randrange(len(seq_list))
            aa = seq_list[idx]
            seq_list[idx] = random.choice([x for x in AMINO_ACIDS if x != aa])
            mutated_prompts.append(" ".join(seq_list))

    print(f"Generando {len(mutated_prompts)} variantes en lotes de {batch_size}...")

    # --- Procesar en lotes ---
    for i in tqdm(range(0, len(mutated_prompts), batch_size), desc="Generando"):
        batch_prompts = mutated_prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt",
                            padding=True, truncation=True, max_length=max_length).to(device)
        max_input_length = max(inputs["input_ids"].shape[1], max_length)
##        --Debug--
##        print(f"\nshape len: {inputs['input_ids'].shape[1]}")
##
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                min_length=min_length,
                max_length=max(min_length+1, max_length, max_input_length+2),
               # max_new_tokens=10,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,                
            )

        # Decodificar batch completo
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        unique_variants.update([s.replace(" ", "") for s in decoded])
        del inputs
        del outputs

    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"\nGeneración completada. Se obtuvieron {len(unique_variants)} variantes únicas.")
    return list(unique_variants)
