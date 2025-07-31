# **Fine-Tuning ESM2 for PHF20 Protein Structure Prediction**

This repository provides a complete implementation for fine-tuning the `facebook/esm2_t12_35M_UR50D` protein language model from Meta AI’s ESM series to predict the 3D structure of the **PHF20** protein. This pipeline focuses on directly learning Cα atomic coordinates from sequence embeddings, offering a lightweight alternative to MSA-based structure prediction.

The repository includes both modular Python scripts and interactive notebooks for reproducibility, benchmarking, and downstream evaluation.


## **About PHF20**

**PHF20 (PHD Finger Protein 20)** is a transcriptional regulator involved in chromatin remodeling, histone modification recognition, and gene expression control. It binds to methylated lysines and plays a critical role in cellular stress response, apoptosis, and cancer progression. Structural insights into PHF20 are essential for understanding its domain organization, binding specificity, and interaction dynamics with chromatin-associated proteins.


## **Methodology**

The fine-tuning strategy implemented here is purely single-sequence based and does not use evolutionary information (e.g., MSAs). The pipeline consists of the following stages:

### **1. Preprocessing**
- Load the experimentally resolved or predicted PHF20 structure (`phf20.pdb`).
- Extract amino acid sequence from a specific chain (default: chain A).
- Retrieve Cα atom coordinates for all residues.
- Convert sequence into FASTA format and save for downstream use.
- Files:  
  - `preprocess_phf20.py` (modular script)  
  - `preprocess_phf20.ipynb` (interactive notebook)

### **2. Embedding Extraction and Model Fine-Tuning**
- Tokenize the FASTA sequence using Hugging Face's `AutoTokenizer`.
- Extract per-residue embeddings using `EsmModel.from_pretrained`.
- Train a simple multi-layer perceptron (MLP) regression head to map embeddings to 3D Cα positions.
- Align predicted coordinates to the reference using the **Kabsch algorithm** for rigid-body RMSD minimization.
- Files:
  - `finetune_phf20.py` (script)
  - `finetune_phf20.ipynb` (notebook)

### **3. Output and Evaluation**
- Predicted coordinates are saved as a new PDB file for downstream visualization or comparison.
- Optional metrics (e.g., per-residue RMSD, trajectory overlays) can be computed and visualized using external tools like PyMOL or MDtraj.
- Output file: `outputs/predicted_structure.pdb`

## **Usage**

### **Environment Setup**

Use the provided `environment.yml` file to install all required dependencies, including PyTorch, Hugging Face Transformers, Biopython, and MDAnalysis tools.

```bash
conda env create -f environment.yml
conda activate esm_phf20_env
