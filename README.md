# Top_FCNC_LHeC_LHmuC

End-to-end study kit for **top-quark FCNC** interactions *(tqγ, tqZ, tqH)* at **lepton–hadron colliders**: **LHeC**, **μLHC** and **FCC-μH**.  
The repo contains ready MadGraph5_aMC process cards, fast analysis scripts, and limit tools to compare channels and colliders on equal footing.

---

## TL;DR

- Generate FCNC single-top in **DIS** (tagged lepton) and **EPA** (γp) modes.  
- Analyze with a fast cut-and-count, then set **95% CL** limits on \( \kappa_{tq\gamma}, \kappa_{tqZ}, \lambda_{tqH} \).  
- Convert coupling limits to **BR(\(t\to qX\))** and compare **LHeC ↔ μLHC ↔ FCC-μH**.

---

## Physics scope

We target the effective interactions
\[
\bar t\,\sigma^{\mu\nu}\,\big(\kappa^\gamma_{tq},\kappa^Z_{tq}\big)\,q\,F_{\mu\nu}/\Lambda, \qquad
\bar t\,\big(\lambda^H_{tq,L}P_L+\lambda^H_{tq,R}P_R\big)\,q\,H,
\]
and study:

- **Production (preferred)**  
  - **DIS:** \( \ell p \to \ell\, t\, j \), \( \ell p \to \ell\, tZ\,j \), \( \ell p \to \ell\, tH\,j \)  
  - **EPA:** \( \gamma p \to t\,j \) *[tqγ]*, \( \gamma p \to tZ\,j \), \( \gamma p \to tH\,j \)
- **Decays (cross-check):** \( t\to q\gamma,\, qZ,\, qH \)

---

## Collider baselines

| Collider | E_ℓ (GeV) | E_p (GeV) | √s (TeV) | Luminosity (pb⁻¹) |
|---|---:|---:|---:|---:|
| LHeC | 60 | 7000 | ≈1.30 | 1.0×10⁶ *(1 ab⁻¹)* |
| μLHC | 500–1000 | 7000 | ≈3.74–5.29 | 1–4×10⁶ |
| FCC-μH | 500–3000 | 50000 | ≈10–27.4 | 5–20×10⁶ |

> Use a **QED-aware proton PDF** (e.g. NNPDF31_nlo_as_0118_luxqed).

---

## Repository layout (suggested)

```
Top_FCNC_LHeC_LHmuC/
├─ cards/
│  ├─ run_card_LHeC_DIS.dat
│  ├─ run_card_muLHC_DIS.dat
│  ├─ run_card_muLHC_EPA.dat
│  ├─ run_card_FCCmuh_DIS.dat
│  └─ run_card_FCCmuh_EPA.dat
├─ processes/              # MG5 process folders (auto-generated)
├─ analysis/
│  ├─ analyze_fcnc_ep.py   # reads MG5 banners → S,B,Z_Asimov
│  └─ scan_coupling.py     # fits σ = A·κ² and returns κ95, BR95
├─ delphes/                # (optional) detector cards / notes
├─ ufo/
│  └─ topFCNC_UFO/         # place the FCNC UFO here
└─ README.md
```

---

## Requirements

- **MadGraph5_aMC** ≥ 3.6  
- **LHAPDF** with a QED set (e.g. ID **325300** = NNPDF31_nlo_as_0118_luxqed)  
- Python 3.9+ (for analysis scripts)

---

## Quick start

1) **Put the UFO** here:
```
ufo/topFCNC_UFO/   # or update the path in your MG5 cards
```

2) **LHeC DIS** *(e⁻p → e⁻tj)* — tqγ/tqZ prototype
```
import model ./ufo/topFCNC_UFO
set complex_mass_scheme True
define p = g u c d s b u~ c~ d~ s~ b~
define j = g u c d s u~ c~ d~ s~

generate e- p > e- t j NP=1
output processes/LHeC_DIS_tqV -f
launch
  ebeam1 = 60.0
  ebeam2 = 7000.0
  lpp1 = 0    # lepton
  lpp2 = 1    # proton
  pdlabel1 = none
  pdlabel2 = lhapdf
  lhaid = 325300
  # turn on one coupling (example: tuγ LH)
  set kappa_tuA_L 0.02
  set kappa_tuA_R 0.0
  set kappa_tcA_L 0.0
  set kappa_tcA_R 0.0
  compute_widths t
  done
```

3) **μLHC DIS** *(μ⁻p → μ⁻tj)*
```
generate mu- p > mu- t j NP=1
launch
  ebeam1 = 500.0    # or 1000.0
  ebeam2 = 7000.0
  lpp1 = 0
  lpp2 = 1
  pdlabel1 = none
  pdlabel2 = lhapdf
  lhaid = 325300
```

4) **μLHC EPA** (γ from μ beam): **γp → tj** — best for tqγ
```
generate a p > t j NP=1
launch
  ebeam1 = 500.0        # muon energy
  ebeam2 = 7000.0
  lpp1 = +4             # muon EPA
  lpp2 = 1              # proton
  pdlabel1 = iww        # EPA model (iww/eva)
  pdlabel2 = lhapdf
  lhaid = 325300
```

5) **FCC-μH**: same cards, set `ebeam1 = 500–3000`, `ebeam2 = 50000`, and extend `|η|` acceptance (e.g. 6.0).

---

## Fast analysis & limits

After each `launch`, MG5 prints the **cross section** in the run banner. The helper script reads those and returns yields/significance.

```
python3 analysis/analyze_fcnc_ep.py
```

For coupling scans (fits σ = A·κ², returns **κ₉₅**):
```
python3 analysis/scan_coupling.py   --mode muLHC_DIS --E1 500 --E2 7000 --LPP1 0 --LPP2 1   --parL kappa_tuA_L --parR kappa_tuA_R   --lumi_pb 2e6 --bkg_pb 0.05
```

> Replace `--mode` and energies to scan LHeC / FCC-μH and tqZ/tqH parameters.

---

## Selections (baseline)

- Exactly one isolated lepton \(p_T>20\) GeV, \(|\eta|<2.5\) (DIS).  
- ≥1 b-tag \(p_T>25\) GeV, \(|\eta|<2.5\).  
- ≥1 forward jet \(p_T>20\) GeV, \(|\eta|>2.5\).  
- Reconstruct \(m_t\) (window 140–210 GeV).  
- For tqH: ≥2 b-tags and \(m_{bb}\in[100,150]\) GeV.

Systematics placeholders: 5–10% on total background; PDF+scale bands; EPA flux model for γp.

---

## Common pitfalls

- **Order keyword:** some UFOs use `NP`, others `FCNC`. Check with `display coupling_orders`. If in doubt, **omit** the order filter first (`generate ...` without `NP=1`) to verify a nonzero σ.  
- **Beam syntax:**  
  - DIS: `lpp1=0, lpp2=1` and the **lepton appears in the process**.  
  - EPA (lepton): `lpp1=+3/+4` and process **starts with** `a p > ...`.  
  - `lpp=2` = **elastic photon from proton**, not lepton EPA.  
- For 2→2 DIS (`ℓ p > ℓ t`), set `ptj=0`. For 2→3 (`ℓ p > ℓ t j`), use `ptj≈20 GeV`.

---

## Roadmap / WPs

- **WP1:** tqγ via **γp→tj** (μLHC/FCC-μH EPA)  
- **WP2:** tqγ in **DIS** (LHeC/μLHC/FCC-μH)  
- **WP3:** tqZ in **DIS**  
- **WP4:** tqH in **DIS** (H→bb)  
- **WP5:** FCNC **decays** \(t	o qX\)  
- **WP6:** Systematics + global combinations

Each WP ships cards, cutflows, κ₉₅/BR₉₅ and a one-page note.

---

## License

Pick a license that fits your collaboration (e.g., MIT for code, CC-BY-4.0 for docs) and add a `LICENSE` file.

---

## Contact

Maintainer: **Hamzeh Khanpour** (AGH University of Kraków).  
Open a GitHub issue for requests/bugs.
