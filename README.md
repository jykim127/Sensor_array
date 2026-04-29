# Sensor_array

Reproducibility package for:

> **Machine Learning Extracts Molecular Design Rules for Colorimetric Chemical Warfare Agent Detection**
> Kim *et al.*, submitted to *J. Am. Chem. Soc.*, 2026.

This repository regenerates every quantitative claim in the manuscript:
RDKit-anchored canonical SMILES, RandomForestClassifier 10-fold cross-validation
(`AUC = 0.880 ± 0.058`), 5-dye prospective holdout
(`AUC = 0.619, accuracy 68.8%`), the physicochemical sweet spot
(`LogP_dye 1.97–4.78, |ΔLogP| 0.78–3.03, ΣTPSA 74.7–125.7 Å²`),
agent-class centroids
(`G 1.87/115.0, V 1.57/108.1, Novichok 1.91/121.3, Vesicants 1.86/80.7`),
and the 6-dye minimal array (D22, D27, D19, D13, D26, D25).

A PyTorch MPNN trained from scratch on the corrected canonical SMILES
(`AUC = 0.852 ± 0.083`, h=96, 3 message-passing rounds, 10-fold CV, seed=42)
is also provided.

---

## Layout

```
.
├── analysis_main.py       — single-shot RandomForest pipeline (Step 4 reproduction)
├── corrected_smiles.py    — CAS-anchored canonical SMILES (28 dyes + 16 CWAs)
├── requirements.txt       — pinned package versions (sklearn 1.7.2, rdkit, etc.)
├── README.md              — this file
├── LICENSE                — MIT for code
├── LICENSE_DATA           — CC-BY-4.0 for the data files
├── CITATION.cff           — citation metadata (GitHub auto-renders)
├── data/
│   ├── comprehensive_analysis_matrix.xlsx — 448-pair feature matrix (response_binary preserved)
│   ├── smiles_database.xlsx               — original SMILES (audit reference)
│   ├── feature_importance.xlsx            — Random Forest Gini importance
│   └── statistical_results.xlsx           — Mann-Whitney U per feature
├── model/
│   ├── cwa_prediction_rfc.pkl   — trained RFC (sklearn 1.7.2, seed=42)
│   └── mpnn_big_final.pt        — trained MPNN state_dict (PyTorch, h=96)
├── scripts/
│   └── mpnn_train.py            — checkpoint-based 10-fold MPNN training
└── results/
    ├── mpnn_cv_metrics.xlsx     — per-fold MPNN AUC/BA/Prec/Rec/F1
    └── mpnn_summary.json        — MPNN summary statistics
```

---

## Quick start

```bash
git clone https://github.com/jykim127/Sensor_array.git
cd Sensor_array

# Optional: virtual environment (Python 3.10–3.12)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Reproduce the Random Forest analysis (~30 s)
python analysis_main.py
```

Console output reproduces every number cited in the paper. `random_state=42`
is fixed throughout, so results are deterministic.

To re-run the MPNN training (requires PyTorch CPU, ~140 s total in checkpoint
mode):

```bash
cd scripts
python mpnn_train.py    # writes results/mpnn_big_cv_metrics.xlsx + model/mpnn_big_final.pt
```

---

## Key results

| Metric | Value | Source |
|--------|------:|:-------|
| 10-fold CV AUC (RandomForest) | 0.880 ± 0.058 | `analysis_main.py` |
| 10-fold CV AUC (MPNN, h=96)   | 0.852 ± 0.083 | `scripts/mpnn_train.py` |
| Holdout AUC (RandomForest)    | 0.619         | `analysis_main.py` |
| Holdout AUC (MPNN)            | 0.469         | `scripts/mpnn_train.py` |
| Holdout accuracy (RF)         | 68.8 % (55/80) | `analysis_main.py` |
| Sweet spot LogP_dye (IQR)     | 1.97 – 4.78   | `analysis_main.py` |
| Sweet spot \|ΔLogP\| (IQR)     | 0.78 – 3.03   | `analysis_main.py` |
| Sweet spot ΣTPSA (IQR, Å²)    | 74.7 – 125.7  | `analysis_main.py` |
| Top 6 dyes by detection breadth | D22, D27, D19, D13, D26, D25 (each 16/16) | `analysis_main.py` |
| Minimal-array coverage        | 100 % of 16 CWAs | `analysis_main.py` |

---

## Notes on data integrity

The **original** 28-dye descriptor table contained inconsistencies between
the published SMILES strings and the reported MW/LogP/TPSA values
(notably a duplicated row between Dye 7 = Fluorescein and
Dye 24 = 4,4′-Dihydroxybenzophenone, an invalid SMILES for Dye 28 =
Rhodamine 6G, and a Fluorescein structure missing one xanthene ring).
`corrected_smiles.py` provides CAS-anchored canonical SMILES for the
28 dyes and 16 authentic CWAs, and `analysis_main.py` recomputes every
descriptor with RDKit (Crippen LogP, topological PSA). The
trained model and every downstream result in the manuscript use these
RDKit-derived descriptors.

The `response_binary` column in `comprehensive_analysis_matrix.xlsx` is
the experimental observation (0/1, threshold rule defined in the
manuscript Methods) and is **unchanged**.

---

## License & citation

| | License |
|---|---|
| Source code (`*.py`) | **MIT** — see `LICENSE` |
| Data files (`data/*.xlsx`) | **CC-BY-4.0** — see `LICENSE_DATA` |
| Trained model objects (`model/*.pkl`, `*.pt`) | **CC-BY-4.0** |

Citation metadata is in `CITATION.cff`; GitHub will auto-render a
"Cite this repository" button.

If you use this code or dataset, please cite the paper as published
(provisional):

> Kim, J.; Yoo, J.; Shin, M.; Kim, S.; Park, J.; Yoo, J.;
> Kim, M.-K.; Lee, D.-H.; Kang, K. *Machine Learning Extracts Molecular
> Design Rules for Colorimetric Chemical Warfare Agent Detection.*
> J. Am. Chem. Soc. **2026**, *XX*, XXXX–XXXX.

---

## Contact

Ku Kang — bisu9082@gmail.com
CBRN Defense Research Institute, Department of Chemistry, Myongji University
