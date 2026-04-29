"""
analysis_main.py
─────────────────
Single-shot end-to-end Step-4 reproducibility script for:
"Machine Learning Extracts Molecular Design Rules for
 Colorimetric Chemical Warfare Agent Detection"

Pipeline:
  1. Recompute RDKit descriptors for 28 dyes + 16 authentic CWAs from
     CAS-anchored canonical SMILES (corrected_smiles.py).
  2. Build the 448-pair feature matrix (response_binary preserved from
     experimental data file).
  3. Hold out 5 dyes (D4 Quinine, D16 Bromophenol Blue, D19 Indigo
     Carmine, D20 Phenol Red, D21 Crystal Violet); train RFC on the
     remaining 23 dyes (368 pairs).
  4. Stratified 10-fold CV with seed=42; report fold-level + mean AUC,
     BA, precision, recall, F1; confusion matrix at theta=0.47 and the
     Youden-optimal threshold; aggregated CV predictions.
  5. Holdout evaluation on 5 dyes x 16 agents = 80 pairs.
  6. Sweet spot quantiles (LogP_dye, |dLogP|, sumTPSA over responsive).
  7. Per-class centroid (G/V/Novichok/Vesicants/Blood/Choking).
  8. Top 6 dyes by detection breadth -> minimal array stats.

Outputs to ./results/. Requires: see requirements.txt (Python 3.10-3.12).
"""
from __future__ import annotations
import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as rd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, accuracy_score, roc_curve)

from corrected_smiles import DYES, AGENTS

SEED = 42
HOLDOUT_DYES = {4, 16, 19, 20, 21}
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

def dye_desc(smi: str) -> dict:
    m = Chem.MolFromSmiles(smi)
    return dict(
        MW=Descriptors.MolWt(m), LogP=Crippen.MolLogP(m),
        TPSA=Descriptors.TPSA(m),
        HBA=rd.CalcNumHBA(m), HBD=rd.CalcNumHBD(m),
        nRotBonds=rd.CalcNumRotatableBonds(m),
        nAromRings=rd.CalcNumAromaticRings(m),
        FormalCharge=Chem.GetFormalCharge(m),
    )

def agent_desc(smi: str) -> dict:
    m = Chem.MolFromSmiles(smi)
    has_PO = int(any(b.GetBondTypeAsDouble() == 2 and
                     {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} == {"P", "O"}
                     for b in m.GetBonds()))
    n_hal = sum(1 for a in m.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))
    n_N = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "N")
    n_S = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "S")
    elec = round(Descriptors.TPSA(m) / Descriptors.MolWt(m) * 100 + has_PO * 5 + n_hal * 1.5, 2)
    return dict(
        MW=Descriptors.MolWt(m), LogP=Crippen.MolLogP(m),
        TPSA=Descriptors.TPSA(m),
        HBA=rd.CalcNumHBA(m), HBD=rd.CalcNumHBD(m),
        nRotBonds=rd.CalcNumRotatableBonds(m),
        has_PO=has_PO, n_halogen=n_hal, n_N=n_N, n_S=n_S,
        electrophilicity=elec,
    )

print(">>> 1. recomputing 28 dye + 16 agent descriptors with RDKit ...")
dye_df = pd.DataFrame([{"dye_num": n, "dye_name": name, "CAS": cas, "SMILES": smi, **dye_desc(smi)}
                       for n, name, cas, smi, _ in DYES])
dye_df["HB_total"] = dye_df["HBA"] + dye_df["HBD"]
agent_df = pd.DataFrame([{"agent_code": code, "agent_name": name, "CAS": cas, "SMILES": smi, **agent_desc(smi)}
                         for code, name, cas, smi in AGENTS])
dye_df.to_excel(RESULTS / "dyes_RDKit_descriptors.xlsx", index=False)
agent_df.to_excel(RESULTS / "agents_RDKit_descriptors.xlsx", index=False)

src = ROOT / "data" / "comprehensive_analysis_matrix.xlsx"
print(f">>> 2. building 448-pair feature matrix from {src.name}")
orig = pd.read_excel(src)
dye_lookup = {r.dye_num: (r.dye_name,
              [r.MW, r.LogP, r.TPSA, r.HBA, r.HBD, r.nRotBonds, r.nAromRings, r.FormalCharge])
              for r in dye_df.itertuples()}
agent_lookup = {r.agent_code: (r.agent_name,
                [r.MW, r.LogP, r.TPSA, r.HBA, r.HBD, r.nRotBonds, r.has_PO, r.n_halogen, r.n_N, r.n_S, r.electrophilicity])
                for r in agent_df.itertuples()}
rows = []
for _, r in orig.iterrows():
    n = int(r.dye_num); code = r.agent_name
    if code not in agent_lookup: continue
    dname, dv = dye_lookup[n]; aname, av = agent_lookup[code]
    rows.append({"dye_num": n, "dye_name": dname, "agent": r.agent, "agent_name": code,
        "dye_MW": dv[0], "dye_LogP": dv[1], "dye_TPSA": dv[2],
        "dye_HBA": dv[3], "dye_HBD": dv[4],
        "dye_nRotBonds": dv[5], "dye_nAromRings": dv[6], "dye_FormalCharge": dv[7],
        "dye_HB_total": dv[3] + dv[4],
        "cwa_MW": av[0], "cwa_LogP": av[1], "cwa_TPSA": av[2],
        "cwa_HBA": av[3], "cwa_HBD": av[4], "cwa_nRotBonds": av[5],
        "cwa_has_PO": av[6], "cwa_n_halogen": av[7],
        "cwa_n_N": av[8], "cwa_n_S": av[9], "cwa_electrophilicity": av[10],
        "delta_LogP": abs(dv[1] - av[1]),
        "sum_TPSA": dv[2] + av[2],
        "HBA_match": int(dv[3] >= av[3]),
        "response_binary": int(r.response_binary),
        "agent_group": r.agent_group})
df = pd.DataFrame(rows)
df.to_excel(RESULTS / "new_analysis_matrix.xlsx", index=False)

exclude = {"dye_num","dye_name","agent","agent_name","response_binary","agent_group"}
feats = [c for c in df.columns if c not in exclude]
train = df[~df.dye_num.isin(HOLDOUT_DYES)].reset_index(drop=True)
hold = df[df.dye_num.isin(HOLDOUT_DYES)].reset_index(drop=True)
X_tr, y_tr = train[feats].values, train.response_binary.astype(int).values
X_h,  y_h  = hold[feats].values,  hold.response_binary.astype(int).values
print(f">>> 3. train n={len(train)}; holdout n={len(hold)}")

print(">>> 4. stratified 10-fold CV (seed=42) ...")
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
folds = []
for i, (tr_i, te_i) in enumerate(skf.split(X_tr, y_tr), 1):
    rf.fit(X_tr[tr_i], y_tr[tr_i])
    p = rf.predict_proba(X_tr[te_i])[:, 1]
    pred = (p >= 0.47).astype(int)
    folds.append({"fold": i, "AUC": roc_auc_score(y_tr[te_i], p),
        "BA": balanced_accuracy_score(y_tr[te_i], pred),
        "Prec": precision_score(y_tr[te_i], pred, zero_division=0),
        "Rec": recall_score(y_tr[te_i], pred, zero_division=0),
        "F1": f1_score(y_tr[te_i], pred, zero_division=0)})
fold_df = pd.DataFrame(folds); fold_df.to_excel(RESULTS / "cv_fold_metrics.xlsx", index=False)
mean = fold_df.mean(numeric_only=True); sd = fold_df.std(numeric_only=True)
print(f"     mean AUC = {mean.AUC:.3f} +/- {sd.AUC:.3f}")
print(f"     mean BA  = {mean.BA:.3f} +/- {sd.BA:.3f}")
print(f"     mean Prec= {mean.Prec:.3f} +/- {sd.Prec:.3f}")
print(f"     mean Rec = {mean.Rec:.3f} +/- {sd.Rec:.3f}")

proba_cv = cross_val_predict(rf, X_tr, y_tr, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
pred47 = (proba_cv >= 0.47).astype(int)
cm = confusion_matrix(y_tr, pred47)
fpr, tpr, thr = roc_curve(y_tr, proba_cv); j = np.argmax(tpr - fpr)
print(f"     CM theta=0.47: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
print(f"       Precision = {cm[1,1]/(cm[1,1]+cm[0,1])*100:.1f}%   Recall = {cm[1,1]/(cm[1,1]+cm[1,0])*100:.1f}%")
print(f"     Youden-optimal theta = {thr[j]:.3f}")

print(">>> 5. holdout evaluation (80 pairs) ...")
rf_final = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
ph = rf_final.predict_proba(X_h)[:, 1]
predh = (ph >= 0.47).astype(int)
auc_h = roc_auc_score(y_h, ph); acc_h = accuracy_score(y_h, predh)
cmh = confusion_matrix(y_h, predh)
print(f"     holdout AUC = {auc_h:.3f}   accuracy = {acc_h*100:.1f}% ({(predh==y_h).sum()}/{len(y_h)})")
print(f"     holdout CM: TN={cmh[0,0]} FP={cmh[0,1]} FN={cmh[1,0]} TP={cmh[1,1]}")
pd.DataFrame({"dye_num": hold.dye_num, "dye_name": hold.dye_name, "agent": hold.agent_name,
              "observed": y_h, "predicted_proba": ph, "predicted_47": predh,
              "correct": (predh == y_h).astype(int)}).to_excel(RESULTS / "holdout_results.xlsx", index=False)
pickle.dump({"model": rf_final, "features": feats, "seed": SEED, "holdout_dyes": sorted(HOLDOUT_DYES)},
            open(RESULTS / "cwa_prediction_model_v2.pkl", "wb"))

print(">>> 6. sweet spot quantiles (responsive only) ...")
resp = df[df.response_binary == 1]
def q(s): return (s.quantile(.25), s.median(), s.quantile(.75))
lp = q(resp.dye_LogP); dl = q(resp.delta_LogP); tp = q(resp.sum_TPSA)
print(f"     dye LogP    : Q1={lp[0]:.2f}  median={lp[1]:.2f}  Q3={lp[2]:.2f}")
print(f"     |dLogP|     : Q1={dl[0]:.2f}  median={dl[1]:.2f}  Q3={dl[2]:.2f}")
print(f"     sumTPSA     : Q1={tp[0]:.1f}   median={tp[1]:.1f}   Q3={tp[2]:.1f}")

print(">>> 7. per-class centroid ...")
centroids = {}
for g in ["G-agents","V-agents","Novichok","Vesicants","Blood","Choking"]:
    sub = resp[resp.agent_group == g]
    if len(sub):
        centroids[g] = (round(sub.delta_LogP.mean(), 2), round(sub.sum_TPSA.mean(), 1))
        print(f"     {g:<10}  dLogP={centroids[g][0]:.2f}  sumTPSA={centroids[g][1]:.1f}")

print(">>> 8. top 6 dyes by detection breadth ...")
breadth = df.groupby("dye_num").agg(name=("dye_name","first"), npos=("response_binary","sum")).sort_values("npos", ascending=False)
top6 = breadth.head(6).index.tolist()
agent_cov = df[df.dye_num.isin(top6)].groupby("agent_name").response_binary.max()
print(f"     top 6 = {top6}   coverage = {int(agent_cov.sum())}/16")

stats = {
    "seed": SEED, "training_pairs": int(len(train)), "holdout_pairs": int(len(hold)),
    "cv_mean_auc": round(float(mean.AUC), 3), "cv_sd_auc": round(float(sd.AUC), 3),
    "cv_mean_BA": round(float(mean.BA), 3), "cv_sd_BA": round(float(sd.BA), 3),
    "cv_mean_Prec": round(float(mean.Prec), 3), "cv_sd_Prec": round(float(sd.Prec), 3),
    "cv_mean_Rec": round(float(mean.Rec), 3), "cv_sd_Rec": round(float(sd.Rec), 3),
    "youden_theta": round(float(thr[j]), 3),
    "holdout_AUC": round(float(auc_h), 3), "holdout_accuracy_pct": round(float(acc_h * 100), 1),
    "sweet_spot": {
        "dye_LogP_IQR": [round(lp[0],2), round(lp[2],2)], "dye_LogP_median": round(lp[1],2),
        "deltaLogP_IQR": [round(dl[0],2), round(dl[2],2)], "deltaLogP_median": round(dl[1],2),
        "sumTPSA_IQR": [round(tp[0],1), round(tp[2],1)], "sumTPSA_median": round(tp[1],1)},
    "centroids": centroids, "top6_dyes": top6,
    "minimal_array_coverage_pct": round(float(agent_cov.mean() * 100), 1)}
json.dump(stats, open(RESULTS / "key_statistics.json", "w"), indent=2)
print(f"\n>>> done. results in: {RESULTS}")
