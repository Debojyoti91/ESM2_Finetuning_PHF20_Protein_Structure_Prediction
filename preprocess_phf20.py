import os
from Bio.PDB import PDBParser, PPBuilder
from Bio import SeqIO
import numpy as np
import pandas as pd
import pickle

def extract_sequence_and_ca_coords(pdb_path, chain_id='A'):
    """
    Extracts amino acid sequence and corresponding alpha carbon (Cα) coordinates
    from a specified chain in a PDB file.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PHF20", pdb_path)
    model = structure[0]
    
    sequence = ""
    coords = []

    for chain in model:
        if chain.id != chain_id:
            continue
        for res in chain:
            if 'CA' in res:
                sequence += res.resname
                coords.append(res['CA'].get_coord())

    coords = np.array(coords)
    return sequence, coords

def convert_three_to_one(seq):
    """
    Converts 3-letter amino acid codes to 1-letter codes.
    """
    from Bio.Data import IUPACData
    table = IUPACData.protein_letters_3to1_extended
    return ''.join([table.get(res.lower().capitalize(), 'X') for res in seq.split()])

def save_fasta(sequence_1letter, output_path):
    """
    Saves a single-sequence FASTA file.
    """
    with open(output_path, 'w') as f:
        f.write(">PHF20_chainA\n")
        f.write(sequence_1letter + "\n")

def main():
    pdb_file = "data/phf20.pdb"
    fasta_file = "data/phf20.fasta"
    output_pickle = "data/processed_embeddings.pkl"

    # Step 1: Extract sequence and Cα coordinates
    sequence_3letter, ca_coords = extract_sequence_and_ca_coords(pdb_file, chain_id='A')
    sequence_1letter = convert_three_to_one(' '.join([sequence_3letter[i:i+3] for i in range(0, len(sequence_3letter), 3)]))

    print(f"Extracted sequence length: {len(sequence_1letter)}")
    print(f"Cα coordinate array shape: {ca_coords.shape}")

    # Step 2: Save FASTA
    save_fasta(sequence_1letter, fasta_file)
    print(f"Saved FASTA to {fasta_file}")

    # Step 3: Save embeddings placeholder (coordinates now, embeddings added later)
    df = pd.DataFrame({
        "sequence": [sequence_1letter],
        "coordinates": [ca_coords]
    })

    with open(output_pickle, 'wb') as f:
        pickle.dump(df, f)

    print(f"Saved coordinate data to {output_pickle}")

if __name__ == "__main__":
    main()

