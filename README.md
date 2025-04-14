# **Fine-Tuning ESM2 for PHF20 Protein Structure Prediction**

This repository contains the implementation for fine-tuning the `esm2_t12_35M_UR50D` model from Meta AI's ESM series, applied to the **PHF20** protein structure. The objective is to improve model fidelity using refined structural information derived from classical and deep learning-based predictions.


## 🔬 About PHF20

**PHF20 (PHD Finger Protein 20)** is a transcriptional regulator implicated in chromatin remodeling and epigenetic gene regulation. It recognizes methylated lysines on histone tails and is functionally associated with cellular stress response, apoptosis, and tumorigenesis. Understanding its 3D structure is crucial for exploring its interaction mechanisms, especially in the context of cancer biology.


## 🧪 Methodology

The overall workflow includes both structure prediction and downstream refinement:

1. **Sequence Retrieval**
   - Protein sequence obtained from **[UniProt](https://www.uniprot.org/)**.

2. **Initial Structure Prediction**
   - Structure predicted using:
     - **ESMFold** (single-sequence prediction without MSA)
     - **AlphaFold2** (with MSA) for higher-fidelity baseline structure

3. **Molecular Dynamics (MD) Simulation**
   - The AlphaFold2-predicted structure was simulated using **GROMACS 2023.1**.
   - A **20 ns MD simulation** was run to evaluate structural stability and physical plausibility.

4. **Model Fine-Tuning**
   - The final stable structure from MD was used to fine-tune the **`esm2_t12_35M_UR50D`** model.
   - Objective: to improve per-residue structural accuracy and align ESM2 predictions with physically-validated ground truth.


## 🚀 Usage

To run the notebook:

1. Open `ESMFold_Fine_tuned_final.ipynb` in **Google Colab**.
2. Follow the cells in sequence:
   - Install dependencies
   - Load sequence and structure
   - Fine-tune ESM2
   - Evaluate predictions
3. Ensure that required input files (e.g., `ref20ns.pdb`) are present in the expected paths. **Soon to be available**


## 📌 Notes

- The notebook is designed for single-sequence fine-tuning using Google Colab GPU.
- The structure used for fine-tuning was derived after stability checks via MD, ensuring robustness in the training target.
- **Soon to be updated for full implementation**


**Feel free to fork or contribute by suggesting improvements or extending to other ESM2 variants or protein systems.**


