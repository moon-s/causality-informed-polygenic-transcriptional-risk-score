# Full Analysis Pipeline: Polygenic Transcriptional Risk Scores from Multi-omics Mendelian Randomization Inform Individualized Disease Risk and Drug Repositioning

_Comprehensive annotated pipeline documentation — generated 2025-10-13 07:07_


---
## Overview

This repository contains the **complete MR-PTRS-Osteo analysis workflow**, integrating **multi-omics Mendelian Randomization (MR)**, **polygenic transcriptional risk scoring (PTRS)**, and **network-based drug prioritization**.

The pipeline processes regulatory variant data (DHS-filtered SNPs), harmonizes GWAS, eQTL, and pQTL datasets, computes causal effect sizes through MR, aggregates them into PTRS, and integrates protein–protein interaction (PPI) networks for module-based and GNN-based drug repositioning.

**Major data resources:**
- GWAS summary statistics for osteonecrosis and metabolic phenotypes  
- eQTL: eQTLGen, GTEx v10  
- pQTL: UK Biobank-PPP, deCODE  
- Regulatory regions: DHS Index and Vocabulary (hg38 WM20190703)  
- Drug–target data: DrugMap 2.0, DGIdb  
- PPI: OmniPath

---
## Pipeline Overview Diagram (simplified)

```
   +-----------------+
   | Regulatory SNPs |  (dbSNP ∩ DHS)
   +--------+--------+
            |
            v
   +-----------------+
   | eQTL / pQTL QC  |  (GWAS/TWAS harmonization)
   +--------+--------+
            |
            v
   +-----------------+
   |   MR Analysis   |  (causal β for genes)
   +--------+--------+
            |
            v
   +-----------------+
   |   β Meta-layer  |  (cross-omics integration)
   +--------+--------+
            |
            v
   +-----------------+
   |  ci-PTRS Scoring|  (β × expression → risk)
   +--------+--------+
            |
            v
   +-----------------+
   |  PPI Network    |  (OmniPath backbone)
   +--------+--------+
            |
            v
   +-----------------+
   |  Causal Modules |  (Leiden, enrichment)
   +--------+--------+
            |
            v
   +-------------------------+
   | Drug Target Enrichment  |
   +--------+----------------+
            |
            v
   +-------------------------+
   | Partial PTRS per Module |
   +--------+----------------+
            |
            v
   +-------------------------+
   | GNN β-Propagation (DRS) |
   +--------+----------------+
            |
            v
   +-------------------------+
   | Figures & Final Outputs |
   +-------------------------+
```

---
## Stage 0–1: Variant and Summary Data Preprocessing

### 0.0 — DHS–dbSNP Filtering  
**Notebook:** `00_DHS-dbSNP_filtering_annotated.ipynb`  
**Purpose:** Filter dbSNP variants within DNase I hypersensitive sites (DHS) and annotate regulatory context.  
**Inputs:** dbSNP VCF (hg38), DHS index.  
**Outputs:** `filtered_dbsnp_in_DHS.tsv`, used for selecting regulatory instruments.  
**Downstream:** Inputs to GWAS/eQTL/pQTL intersection scripts.

### 0.1 — GWAS/TWAS Preprocessing  
**Notebook:** `01_GWAS-TWAS_preprocessing_annotated.ipynb`  
**Purpose:** Harmonize GWAS and TWAS summary statistics: allele alignment, reference matching, and liftover validation.  
**Outputs:** Harmonized GWAS tables ready for MR.  
**Downstream:** MR input (1_1_running_MR).

### 0.2 — pQTL (UK Biobank-PPP) Processing  
**Notebook:** `02_pQTL_UKBB_processing_annotated.ipynb`  
**Purpose:** Parse UK Biobank pQTL summary data, map to protein–gene identifiers, and apply QC.  
**Outputs:** `ukbb_pqtl_ready.tsv`.  
**Downstream:** Input for MR causal inference.

### 0.3 — pQTL (deCODE) Processing  
**Notebook:** `03_pQTL_deCODE_processing_annotated.ipynb`  
**Purpose:** Standardize and merge deCODE pQTL data for cross-cohort meta-analysis.  
**Outputs:** `decode_pqtl_ready.tsv`.  
**Downstream:** Combined MR exposure dataset.

### 0.4 — SNP Statistics and QC  
**Notebook:** `04_SNP_statistics_QC_annotated.ipynb`  
**Purpose:** Compute descriptive statistics (MAF, INFO, HWE) and QC summaries of the SNP universe.  
**Outputs:** `snp_qc_summary.tsv`.  
**Downstream:** Validation of instruments and MR sensitivity.

---
## Stage 1–2: Mendelian Randomization and Effect Integration

### 1.1 — Running MR  
**Notebook:** `11_MR_analysis_core_annotated.ipynb`  
**Purpose:** Run TwoSampleMR (IVW, Egger, weighted median) for eQTL/pQTL exposures → disease outcomes.  
**Outputs:** Per-gene causal β, p-value, SE tables.  
**Downstream:** β meta-integration.

### 1.2 — β Meta-Integration (Cross-Omics)  
**Notebook:** `12_Beta_meta-integration_cross-omics_annotated.ipynb`  
**Purpose:** Integrate MR effect sizes across transcriptomic/proteomic layers to derive unified causal weights.  
**Outputs:** `meta_beta_per_gene.tsv`.  
**Downstream:** Input for PTRS computation.

### 2.1 — MR Visualization (R)  
**Notebook:** `21_MR_results_visualization_R_annotated.ipynb`  
**Purpose:** Generate forest, scatter, and funnel plots for top causal genes; visualize sensitivity tests.  
**Outputs:** Figures (`/results_figures/MR_*`).  
**Downstream:** Manuscript Figure panels.

---
## Stage 3: Polygenic Transcriptional Risk Scoring (ci-PTRS)

### 3.1 — ci-PTRS from Bulk Expression  
**Notebook:** `31_ci-PTRS_bulk_expression_scoring_annotated.ipynb`  
**Purpose:** Multiply normalized expression (GSE123568 or equivalent) by causal β to compute individual PTRS values.  
**Outputs:** `ci_ptrs_per_individual.tsv`; group-level distributions.  
**Downstream:** Integration with PPI modules and DRS validation.

---
## Stage 5: Network Construction and Causal Module Detection

### 5.0 — PPI OmniPath Processing  
**Notebook:** `50_PPI_OmniPath_processing_annotated.ipynb`  
**Purpose:** Build PPI network from OmniPath; map genes, remove self-loops, compute edge weights, export GraphML.  
**Outputs:** `ppi_omnipath.graphml`.  
**Downstream:** Input for Leiden module detection.

### 5.1 — Causal Module Discovery (Leiden)  
**Notebook:** `51_Causal_modules_Leiden_on_PPI_annotated.ipynb`  
**Purpose:** Perform Leiden/CPM community detection on causal subnetwork; grid-search resolution parameters.  
**Outputs:** Module assignments, resolution–enrichment profiles.  

---
## Stage 6: Drug Enrichment, Partial PTRS, and GNN Prioritization

### 6.0 — Drug Target Enrichment Analysis  
**Notebook:** `60_Drug_target_enrichment_analysis_annotated.ipynb`  
**Purpose:** Intersect module genes with DrugMap 2.0 targets; compute hypergeometric enrichment per drug and globally.  
**Outputs:** `drug_module_enrichment.tsv`.  
**Downstream:** Basis for partial PTRS and DRS estimation.

### 6.1 — Partial PTRS by Module  
**Notebook:** `61_Partial_PTRS_by_module_annotated.ipynb`  
**Purpose:** Compute ci-PTRS restricted to each module; compare between control vs disease groups with significance marks.  
**Outputs:** Heatmaps, `partial_ptrs_module.tsv`.  
**Downstream:** GNN input and figure panels.

### 6.2 — Drug Prioritization via GNN  
**Notebook:** `62_Drug_prioritization_GNN_annotated.ipynb`  
**Purpose:** Train Graph Neural Networks (GCN/GAT) over causal modules to propagate MR β and predict drug–gene responses.  
**Outputs:** Predicted β matrices, feature embeddings, DRS per individual.  
**Downstream:** Network propagation and prioritization.

### 6.2a — GNN from GraphML Usage Guide  
**Notebook:** `62a_GNN_from_GraphML_usage_annotated.ipynb`  
**Purpose:** Provides examples of how to use the script `gnn_from_graphml.py` for β propagation training and evaluation.  
**Outputs:** Documentation notebook only.  
**Downstream:** Supports 6.2 workflows.

### Supporting Script — GNN Module  
**File:** `gnn_from_graphml.py`  
**Purpose:** Implements `train_from_graphml()` for GCN/GAT-based β regression from GraphML networks.  
**Inputs:** GraphML files from Step 5.0–5.2.  
**Outputs:** Predicted β per node, saved to `/gnn_from_graphml_runs_submodules/`.

---
## etc. 

### 9.0 — Miscellaneous Figures and Panels  
**Notebook:** `90_Misc_figures_and_panels_annotated.ipynb`  
**Purpose:** Aggregate and format final plots for manuscript (heatmaps, schematic panels, module summaries).  
**Outputs:** `/results_figures/final_panels/`.  
**Downstream:** Manuscript integration.

### 9.1 — Sensitivity analysis
**Notebook:** `91_Sensitivity_analysis.ipynb`  
**Purpose:** Addtional QC for instrumental variable and MR results.  
**Outputs:** `/MR_ready/sensitivity_analysis/`.  
**Downstream:** Supplementary information.  

---
## Reproducibility and Environment

- **Programming languages:** Python ≥3.10, R ≥4.3  
- **Key packages:** `pandas`, `numpy`, `networkx`, `torch-geometric`, `igraph`, `TwoSampleMR`, `ggplot2`, `seaborn`, `statsmodels`  
- **Version control:** All intermediate tables saved under `/results_*` folders with consistent naming.  
- **Logging:** Each notebook automatically detects and records I/O paths.  
- **Random seeds:** Fix seeds for GNN training (`torch.manual_seed`, `numpy.random.seed`).  
- **Compute environment:** Reproducible across Linux (HPC or WSL) with sufficient memory (≥64GB recommended).

---
## Citation and Attribution

This pipeline is part of the **MR-PTRS-Osteo** project, integrating multi-omics causal inference and network pharmacology for individualized disease risk estimation and drug repositioning.

When using or adapting this workflow, please cite:  
**“Polygenic Transcriptional Risk Scores from Multi-omics Mendelian Randomization Inform Individualized Disease Risk and Drug Repositioning.”**

---
