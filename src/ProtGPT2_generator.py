import torch
import random
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # <--- CAMBIO: Importar pipeline
from accelerate import Accelerator

# Lista estándar de aminoácidos



def generate_with_protgpt2_pipeline(
    prompt_sequences: list,
    model_name_or_path: str = "nferruz/ProtGPT2",    
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

    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Usando dispositivo: {device}")

    # Configurar el tokenizer primero ---
    # Lo cargamos por separado para poder ajustar sus propiedades especiales
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left', eos_token_id=0, pad_token_id=0)
    tokenizer.eos_token_id = 0
    tokenizer.pad_token = tokenizer.eos_token
    try:
        # Inicializar el pipeline
        # Pasamos el modelo, el tokenizer ya configurado y el dispositivo.
        print("Cargando pipeline de generación...")
        generator = pipeline(
            "text-generation",
            model=model_name_or_path,
            tokenizer=tokenizer,
            device=device
        )
        print("Pipeline cargado.")

        unique_variants = set()
        
        # PreGenerar todas las mutaciones (Esta lógica se mantiene igual que con proxlnet)
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
                    # (Opcional) Los prints de debug se pueden quitar para más velocidad
                    # print(f"Secuencia truncada a partir del índice {cut_idx}: {base_seq}")
                    # print(f"secuencia original: {base_seq}, secuencia truncada: {sequence_to_mutate} ")

                seq_list = list(sequence_to_mutate)
                idx = random.randrange(len(seq_list))
                aa = seq_list[idx]
                seq_list[idx] = random.choice([x for x in AMINO_ACIDS if x != aa])
                mutated_prompts.add("".join(seq_list))

        print(f"Generando {len(mutated_prompts)} variantes (el pipeline manejará los lotes)...")

        # Ajustar prompts para ProtGPT2
        formatted_prompts = [f"<|endoftext|>\n{seq}" for seq in mutated_prompts]

        # Nos pasamos todos los parámetros de generate() directamente al pipeline.
        gen_kwargs = dict(
            min_length=min_length,
            max_length=max_length,         # El pipeline maneja esto como longitud total
            max_new_tokens=max_new_tokens, # Tu parámetro original
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id, # Importante pasarlo o explota
            repetition_penalty=repetition_penalty,
            # El pipeline no trunca automáticamente los prompts largos
        )

        # El pipeline nos mostrará su propia barra de progreso si tqdm está instalado
        outputs = generator(
            formatted_prompts, 
            batch_size=batch_size, 
            **gen_kwargs
        )

        # La salida es: List[List[Dict[str, str]]]
        # Lista externa: un elemento por cada prompt de entrada
        # Lista interna: un elemento por cada num_return_sequences
        # Diccionario: {'generated_text': '...'}
        
        print("Procesando salidas...")
        for prompt_outputs in tqdm(outputs, desc="Procesando"):
            for seq_dict in prompt_outputs: # Iterar sobre las num_return_sequences
                generated_text = seq_dict['generated_text']
                
                # Tu lógica de limpieza original
                clean = generated_text.replace(" ", "").replace("<|endoftext|>", "").replace("\n", "").strip()
                
                # Tu lógica de truncamiento final
                if len(clean) > max_length:
                    clean = clean[:max_length]
                if len(clean) < min_length:
                    continue
                unique_variants.add(clean)

        # Limpieza de memoria
    finally:
        # Limpieza de memoria
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if generator is not None:
            del generator # El pipeline contiene el modelo no se debe eliminar por separado
        if tokenizer is not None: del tokenizer
        if accelerator is not None: del accelerator
    
    print(f"\nGeneración completada. Se obtuvieron {len(unique_variants)} variantes únicas.")
    return list(unique_variants)