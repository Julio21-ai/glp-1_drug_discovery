# Requisitos: biopython
# pip install biopython

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
#from Bio.SubsMat.MatrixInfo import blosum62
from Bio.Align.substitution_matrices import load

# Load the BLOSUM62 substitution matrix
blosum62 = load("BLOSUM62")

import math

def load_alignment(path):
    aln = AlignIO.read(path, "clustal")
    return aln

def find_ref_record(aln, ref_id):
    for rec in aln:
        if rec.id == ref_id or rec.name == ref_id:
            return rec
    # intentar búsqueda por substring
    for rec in aln:
        if ref_id in rec.id:
            return rec
    raise KeyError(f"No se encontró el registro {ref_id} en el alineamiento")

def ungapped(seq):
    return str(seq).replace("-", "")

def find_peptide_in_unaligned(ref_seq, peptide):
    idx = ref_seq.find(peptide)
    if idx == -1:
        return None
    return idx, idx + len(peptide)  # posiciones 0-based en la secuencia sin huecos

def map_ungapped_coords_to_alignment_indices(aln_ref_seq, ungapped_start, ungapped_end):
    """
    Dado el string alineado (con '-') del registro de referencia,
    y las coordenadas en la secuencia sin huecos [start,end),
    devuelve la lista de índices de columnas del alineamiento que corresponden.
    """
    indices = []
    ungapped_pos = 0
    for aln_idx, res in enumerate(str(aln_ref_seq)):
        if res != "-":
            # estamos en posición ungapped_pos de la secuencia real
            if ungapped_start <= ungapped_pos < ungapped_end:
                indices.append(aln_idx)
            ungapped_pos += 1
    # validar longitud
    if len(indices) != (ungapped_end - ungapped_start):
        # podría ocurrir si no encontró (por ej. errores)
        pass
    return indices

def extract_region_from_alignment(aln, indices):
    """Devuelve diccionario id -> region_string (con huecos si los hay en la secuencia)."""
    out = {}
    for rec in aln:
        seq = str(rec.seq)
        region = "".join(seq[i] for i in indices)
        out[rec.id] = region
    return out

def extract_region(aln, indices):
    regions = {}
    for rec in aln:
        seq = str(rec.seq)
        region = "".join(seq[i] for i in indices)
        regions[rec.id] = region
    return regions

def percent_identity(seqA, seqB):
    """Ambas strings deben tener misma longitud."""
    assert len(seqA) == len(seqB)
    matches = sum(1 for a,b in zip(seqA,seqB) if a == b and a != "-")
    # definimos longitud efectiva sin contar posiciones donde referencia tiene '-'?
    # aquí contamos posiciones donde la referencia NO es '-'
    effective_len = sum(1 for a in seqA if a != "-")
    if effective_len == 0:
        return 0.0
    return 100.0 * matches / effective_len

def blosum62_score(seqA, seqB, matrix=blosum62):
    """Suma de puntuaciones BLOSUM62 (salta posiciones con '-')"""
    score = 0
    count = 0
    for a,b in zip(seqA, seqB):
        if a == "-" or b == "-":
            continue
        # matrix guarda tuplas con orden (a,b) ó (b,a); manejar KeyError
        key = (a.upper(), b.upper())
        if key in matrix:
            score += matrix[key]
        else:
            key2 = (b.upper(), a.upper())
            if key2 in matrix:
                score += matrix[key2]
            else:
                # residuo raro (p.ej. X), aplicar penalización ligera:
                score += -1
        count += 1
    return score, count

def analyze_peptide_candidates(aln_path, ref_id, peptide, verbose=True):
    aln = load_alignment(aln_path)
    ref = find_ref_record(aln, ref_id)
    ref_ungapped = ungapped(ref.seq)

    pos = find_peptide_in_unaligned(ref_ungapped, peptide)
    if pos is None:
        print("No se encontró la subsecuencia EXACTA en la secuencia sin huecos del registro de referencia.")
        # sugerencia: podrías buscar con mismatches usando pairwise2 si esto ocurre.
        return

    start, end = pos
    print(f"Péptido encontrado en la secuencia sin huecos de la referencia en posiciones {start}..{end-1} (0-based). Longitud {end-start} aa")

    aln_indices = map_ungapped_coords_to_alignment_indices(ref.seq, start, end)
    print(f"Se corresponde con {len(aln_indices)} columnas del alineamiento (índices {aln_indices[0]}..{aln_indices[-1]})")

    regions = extract_region_from_alignment(aln, aln_indices)
    ref_region = regions[ref.id]  # la región alineada para la referencia (incluye '-' si corresponde)
    results = []
    for rec_id, region in regions.items():
        if rec_id == ref.id:
            continue
        pid = percent_identity(ref_region, region)
        bscore, count = blosum62_score(ref_region, region)
        # normalizar score por longitud útil
        norm_bscore = bscore / count if count>0 else float("-inf")
        exact = (region.replace("-","") == peptide)
        results.append({
            "id": rec_id,
            "region": region,
            "percent_identity": pid,
            "blosum62_score": bscore,
            "blosum62_norm": norm_bscore,
            "exact_match": exact,
            "positions_compared": count
        })

    # ordenar por percent_identity luego por blosum62_norm
    results_sorted = sorted(results, key=lambda r: (r["percent_identity"], r["blosum62_norm"]), reverse=True)

    if verbose:
        # imprimir resumen
        print("\nTop candidatos:")
        for r in results_sorted[:30]:
            tag = "EXACTO" if r["exact_match"] else ""
            print(f"{r['id']:30s} ID%: {r['percent_identity']:6.2f}  B62_norm:{r['blosum62_norm']:+5.2f}  compared:{r['positions_compared']:2d}  {tag}")

    return results_sorted


def check_critical_residues(ref_region, seq_region, critical_positions :list):
    status = {}
    for pos in critical_positions:
        idx = pos - 1  # convertir a 0-based
        if idx >= len(ref_region) or seq_region[idx] == "-":
            status[pos] = "?"
        elif seq_region[idx] == ref_region[idx]:
            status[pos] = "Y"
        else:
            status[pos] = "N"
    return status


def analyze_peptide_extended(aln_path, ref_id, peptide, critical_positions :list, verbose=True):
    aln = AlignIO.read(aln_path, "clustal")
    ref = None
    for rec in aln:
        if rec.id == ref_id or ref_id in rec.id:
            ref = rec
            break
    if ref is None:
        raise KeyError(f"No se encontró el registro {ref_id}")
    
    ref_ungapped = ungapped(ref.seq)
    pos = find_peptide_in_unaligned(ref_ungapped, peptide)
    if pos is None:
        print("No se encontró el péptido en la secuencia sin gaps de la referencia.")
        return
    
    start, end = pos
    aln_indices = map_ungapped_coords_to_alignment_indices(ref.seq, start, end)
    regions = extract_region(aln, aln_indices)
    ref_region = regions[ref.id]

    results = []
    for rec_id, region in regions.items():
        if rec_id == ref.id: continue
        pid = percent_identity(ref_region, region)
        bscore, count = blosum62_score(ref_region, region)
        norm_score = bscore / count if count>0 else float("-inf")
        critical_status = check_critical_residues(ref_region, region, critical_positions)
        results.append({
            "id": rec_id,
            "region": region,
            "percent_identity": pid,
            "blosum62_norm": norm_score,
            "critical": critical_status
        })

    results_sorted = sorted(results, key=lambda r: (r["percent_identity"], r["blosum62_norm"]), reverse=True)

    if verbose:
        # imprimir tabla resumida
        print(f"{'ID':30s} {'ID%':>6s} {'B62':>6s} {'Criticos':>20s} Region")
        for r in results_sorted:
            crit_str = "".join(r["critical"][p] for p in critical_positions)
            print(f"{r['id']:30s} {r['percent_identity']:6.1f} {r['blosum62_norm']:6.2f} {crit_str:>20s} {r['region']}")
        
    return results_sorted