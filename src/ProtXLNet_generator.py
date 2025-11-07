# peptide_generator.py

import random
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import XLNetTokenizer, XLNetLMHeadModel, pipeline
from accelerate import Accelerator

# Lista estándar de aminoácidos
AMINO_ACIDS_STRING = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACIDS = list(AMINO_ACIDS_STRING)

pattern_to_remove = f"[^{AMINO_ACIDS_STRING}]"



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

import random
import torch
import re  # <--- CAMBIO: Importar el módulo de expresiones regulares
from tqdm.auto import tqdm
from transformers import XLNetTokenizer, XLNetLMHeadModel, pipeline
from accelerate import Accelerator

# Lista estándar de aminoácidos
AMINO_ACIDS_STRING = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACIDS = list(AMINO_ACIDS_STRING)
# El patrón de regex para eliminar cualquier cosa que NO sea un AA
pattern_to_remove = f"[^{AMINO_ACIDS_STRING}]"


def generate_with_protxlnet_pipeline(
    ruta_modelo_protxlnet: str,
    prompt_sequences: list,
    num_variants_per_seq: int = 5,
    num_return_sequences: int = 5,
    min_length: int = 25,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 3,
    top_p: float = 0.95,
    max_new_tokens: int = 10,
    repetition_penalty: float = 1.0,
    batch_size: int = 32,
    apply_truncation: bool = True,
    truncation_prob: float = 0.3,
    start_cut_pos: int = 5,
) -> list:

    accelerator = Accelerator()
    device = accelerator.device
    print(f"Usando dispositivo: {device}")

    # CAMBIO: Inicializar a None para el bloque 'finally'
    tokenizer = None
    generator = None

    try: 
        # --- CAMBIO: Configurar el tokenizer (forma más limpia) ---
        print("Cargando tokenizer de ProtXLNet...")
        tokenizer = XLNetTokenizer.from_pretrained(
            ruta_modelo_protxlnet,
            # No pasamos max/min length aquí
        )

        # Configuración ESENCIAL para generación: padding a la izquierda
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token        
        
        # --- Cargar pipeline ---
        print("Cargando pipeline de generación...")
        generator = pipeline(
            "text-generation",
            # El modelo se carga aquí
            model=XLNetLMHeadModel.from_pretrained(ruta_modelo_protxlnet),
            tokenizer=tokenizer, # Pasamos el tokenizer ya configurado
            device=device
        )
        print("Pipeline cargado.")

        unique_variants = set()
    
        print("Preparando las secuencias iniciales...")
        mutated_prompts = set()
        for i in tqdm(range(len(prompt_sequences))):
            base_seq = prompt_sequences[i]
            if not base_seq:
                continue

            for _ in range(num_variants_per_seq):
                sequence_to_mutate = base_seq
                if apply_truncation and random.random() < truncation_prob and len(base_seq) > 1:
                    cut_idx = random.randint(min(start_cut_pos, len(base_seq) - 2), len(base_seq) - 2)
                    sequence_to_mutate = base_seq[0:cut_idx]

                seq_list = list(sequence_to_mutate)
                idx = random.randrange(len(seq_list))
                aa = seq_list[idx]
                seq_list[idx] = random.choice([x for x in AMINO_ACIDS if x != aa])
                
                mutated_prompts.add(" ".join(seq_list)) 

        print(f"Generando {len(mutated_prompts)} variantes (el pipeline manejará los lotes)...")

        formatted_prompts = list(mutated_prompts)

        gen_kwargs = dict(
            min_length=min_length,
            max_length=max_length,         
            max_new_tokens=max_new_tokens, 
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id, 
            repetition_penalty=repetition_penalty,
        )

        outputs = generator(
            formatted_prompts, 
            batch_size=batch_size, 
            **gen_kwargs
        )

        print("Procesando salidas...")
        for prompt_outputs in tqdm(outputs, desc="Procesando"):
            for seq_dict in prompt_outputs:
                generated_text = seq_dict['generated_text']
                
                # 1. Quitar espacios y saltos de línea
                no_spaces_text = generated_text.replace(" ", "").replace("\n", "").strip()
                # 2. Usar regex para quitar CUALQUIER OTRA COSA que no sea un AA
                clean = re.sub(pattern_to_remove, '', no_spaces_text)
                
                # Truncamiento final (tu lógica es correcta)
                if len(clean) > max_length:
                    clean = clean[:max_length]
                if len(clean) < min_length:
                    continue                    
                    
                unique_variants.add(clean)

    finally:
        # --- 5. Limpieza de memoria ---
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Usamos los 'if' para evitar errores si algo falló al cargar
        if generator is not None:
            del generator
        if tokenizer is not None:
            del tokenizer
        if accelerator is not None: 
            del accelerator
    
    print(f"\nGeneración completada. Se obtuvieron {len(unique_variants)} variantes únicas.")
    return list(unique_variants)