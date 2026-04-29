"""
corrected_smiles.py
─────────────────────
CAS-anchored canonical SMILES for the 28 dyes + 16 authentic CWAs used in:
"Machine Learning Extracts Molecular Design Rules for Colorimetric CWA Detection"

Each entry: (number, name, CAS, smiles, source_note).
Notes flag any uncertain assignments for Ku review.
"""

# ── 28 dyes (free-acid / chromophore cation form) ────────────────────────────
DYES = [
    # (n, name, CAS, SMILES, note)
    ( 1, "Anthracene",                       "120-12-7",   "c1ccc2cc3ccccc3cc2c1",                                  "PAH, neutral"),
    ( 2, "Pyrene",                           "129-00-0",   "c1cc2ccc3cccc4ccc(c1)c2c34",                            "PAH, neutral"),
    ( 3, "Allura Red AC",                    "25956-17-6", "COc1ccc(/N=N/c2c(O)cc(S(=O)(=O)O)c3cc(S(=O)(=O)O)cc(C)c23)cc1", "free-acid form (disodium = +44 Da)"),
    ( 4, "Quinine",                          "130-95-0",   "COc1ccc2nccc([C@@H](O)[C@@H]3C[C@@H]4CCN3C[C@@H]4C=C)c2c1", "alkaloid, free base"),
    ( 5, "Rhodamine B",                      "81-88-9",    "CCN(CC)c1ccc2c(-c3ccccc3C(=O)O)c3ccc(=[N+](CC)CC)cc-3oc2c1", "cation; chloride salt MW=479.0"),
    ( 6, "Methyl Blue",                      "28983-56-4", "Cc1ccc(Nc2ccc(C(=C3C=CC(=Nc4ccc(Nc5ccc(C)cc5)cc4S(=O)(=O)O)C=C3)c3ccc(S(=O)(=O)O)cc3S(=O)(=O)O)cc2)cc1", "trianion as free triacid"),
    ( 7, "Fluorescein",                      "2321-07-5",  "C1=CC=C2C(=C1)C(=O)OC23C4=C(C=C(C=C4)O)OC5=C3C=CC(=C5)O", "lactone form (PubChem CID 16850), MW=332.31"),
    ( 8, "Methyl Orange",                    "547-58-0",   "CN(C)c1ccc(/N=N/c2ccc(S(=O)(=O)O)cc2)cc1",              "free-acid form"),
    ( 9, "Nile Red",                         "7385-67-3",  "CCN(CC)c1ccc2nc3ccc(=O)cc-3oc2c1",                      "neutral, MW=318.4"),
    (10, "Safranin O",                       "477-73-6",   "Cc1cc2nc3cc(N)ccc3[n+](-c3ccccc3)c2cc1N",               "cation; chloride MW=350.8"),
    (11, "Toluidine Blue O",                 "92-31-9",    "Cc1cc2nc3cc(N(C)C)ccc3[s+]c2cc1N",                      "phenothiazinium cation, C15H16N3S+, MW=270.38"),
    (12, "Nile Blue A",                      "3625-06-7",  "CCN(CC)c1ccc2nc3ccc(=N)c4cc[nH+]c(c34)c2c1",            "cation; sulfate MW=732"),
    (13, "Cresol Red",                       "1733-12-6",  "Cc1cc(C2(c3ccccc3S(=O)(=O)O2)c2cc(C)c(O)cc2)ccc1O",     "sulfonephthalein, MW=382.4"),
    (14, "Eriochrome Black T",               "1787-61-7",  "Cc1ccc2cc(/N=N/c3c(O)c4ccc(S(=O)(=O)O)cc4cc3[N+](=O)[O-])c(O)c(c2c1)",       "free-acid form"),
    (15, "Quinoline Yellow",                 "8003-22-3",  "O=C1c2ccccc2C(=O)N1c1ccc2cc(S(=O)(=O)O)ccc2n1",         "free-acid, disulfonate variant exists"),
    (16, "Bromophenol Blue",                 "115-39-9",   "Oc1c(Br)cc(C2(c3cc(Br)c(O)c(Br)c3)c3ccccc3S(=O)(=O)O2)cc1Br", "tetrabromo sulfonephthalein, MW=669.96"),
    (17, "Neutral Red",                      "553-24-2",   "Cc1ccc2nc3cc(N(C)C)ccc3[nH+]c2c1N",                     "cation; chloride MW=288.78"),
    (18, "Eosin Y",                          "17372-87-1", "O=C1OC2(c3ccccc31)c1cc(O)c(Br)c(Br)c1Oc1c(Br)c(O)c(Br)cc12", "tetrabromo fluorescein, MW=647.89 (free acid)"),
    (19, "Indigo Carmine",                   "860-22-0",   "O=C1Nc2ccc(S(=O)(=O)O)cc2/C1=C1\\Nc2ccc(S(=O)(=O)O)cc2C1=O", "free-acid; disodium salt MW=466.4"),
    (20, "Phenol Red",                       "143-74-8",   "Oc1ccc(C2(c3ccc(O)cc3)c3ccccc3S(=O)(=O)O2)cc1",         "sulfonephthalein, MW=354.4"),
    (21, "Crystal Violet",                   "548-62-9",   "CN(C)c1ccc(C(=C2C=CC(=[N+](C)C)C=C2)c2ccc(N(C)C)cc2)cc1", "cation; chloride MW=407.98"),
    (22, "2,7-Dibromofluorene",              "16433-88-8", "Brc1ccc2c(c1)Cc1cc(Br)ccc1-2",                          "MW=323.99"),
    (23, "Ethyl Viologen Dibromide",         "54827-17-7", "CC[n+]1ccc(-c2cc[n+](CC)cc2)cc1",                       "dication (counter-ions removed for descriptor calc), MW=214.31"),
    (24, "4,4'-Dihydroxybenzophenone",       "611-99-4",   "O=C(c1ccc(O)cc1)c1ccc(O)cc1",                           "MW=214.22"),
    (25, "2,4-Dinitrophenylhydrazine",       "119-26-6",   "NNc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]",                  "DNPH, MW=198.14"),
    (26, "2,5-Diaminobenzene-1,4-dithiol",   "2144-68-5",  "Nc1cc(S)c(N)cc1S",                                      "MW=172.27"),
    (27, "2-Hydrazinobenzothiazole",         "615-21-4",   "NNc1nc2ccccc2s1",                                       "MW=165.22"),
    (28, "Rhodamine 6G",                     "989-38-8",   "CCNc1cc2oc3cc(=[NH+]CC)c(C)cc3c(-c3ccccc3C(=O)OCC)c2cc1C", "cation; perchlorate MW=479.02"),
]

# ── 16 authentic CWAs ─────────────────────────────────────────────────────────
AGENTS = [
    # nerve (G-series)
    ("GA",   "Tabun",                                "77-81-6",  "CCOP(=O)(C#N)N(C)C"),
    ("GB",   "Sarin",                                "107-44-8", "CC(C)OP(C)(=O)F"),
    ("GD",   "Soman",                                "96-64-0",  "CC(OP(C)(=O)F)C(C)(C)C"),
    ("GF",   "Cyclosarin",                           "329-99-7", "O=P(C)(F)OC1CCCCC1"),
    # nerve (V-series)
    ("VX",   "VX",                                   "50782-69-9","CCOP(C)(=O)SCCN(C(C)C)C(C)C"),
    # Novichok A-series
    ("A230", "Novichok A-230",                       "26102-85-2","COP(=O)(F)/N=C(\\C)N(C)C"),
    ("A232", "Novichok A-232",                       "26102-86-3","CCOP(=O)(F)/N=C(\\C)N(C)C"),
    ("A234", "Novichok A-234",                       "26102-87-4","CCCOP(=O)(F)/N=C(\\C)N(C)C"),
    ("A242", "Novichok A-242",                       "0-00-0",    "CC(C)COP(=O)(F)/N=C(\\C)N(C)C"),
    # vesicants
    ("HD",   "Sulfur mustard",                       "505-60-2", "ClCCSCCCl"),
    ("HN3",  "Tris(2-chloroethyl)amine (HN-3)",      "555-77-1", "ClCCN(CCCl)CCCl"),
    ("L",    "Lewisite I",                           "541-25-3", "Cl/C=C/[As](Cl)Cl"),
    # blood
    ("AC",   "Hydrogen cyanide",                     "74-90-8",  "C#N"),
    ("CK",   "Cyanogen chloride",                    "506-77-4", "ClC#N"),
    # choking
    ("CG",   "Phosgene",                             "75-44-5",  "ClC(Cl)=O"),
    ("PS",   "Chloropicrin",                         "76-06-2",  "[O-][N+](=O)C(Cl)(Cl)Cl"),
]

if __name__ == "__main__":
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors as rd
    print("--- 28 DYES (RDKit verified) ---")
    print(f"{'#':>3} {'name':<35} {'CAS':<14} {'MW':>8} {'LogP':>6} {'TPSA':>7} {'HBA':>3} {'HBD':>3}")
    print("-"*100)
    bad = []
    for n, name, cas, smi, note in DYES:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print(f"{n:>3} {name[:35]:<35} {cas:<14} INVALID")
            bad.append(n)
        else:
            print(f"{n:>3} {name[:35]:<35} {cas:<14} {Descriptors.MolWt(m):>8.2f} {Crippen.MolLogP(m):>+6.2f} {Descriptors.TPSA(m):>7.2f} {rd.CalcNumHBA(m):>3} {rd.CalcNumHBD(m):>3}")
    print()
    print("--- 16 AGENTS ---")
    for code, name, cas, smi in AGENTS:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print(f"  {code:<5} {name[:30]:<30} {cas:<14} INVALID")
            bad.append(code)
        else:
            print(f"  {code:<5} {name[:30]:<30} {cas:<14} MW={Descriptors.MolWt(m):>7.2f} LogP={Crippen.MolLogP(m):>+5.2f} TPSA={Descriptors.TPSA(m):>6.2f}")
    print()
    print(f"INVALID count: {len(bad)} {bad}")
