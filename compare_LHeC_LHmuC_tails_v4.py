#!/usr/bin/env python3



"""

python3 compare_LHeC_LHmuC_tails_v4.py \
  --lhec-signal  /home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHeC/Events/run_01/singletop_FCNC_LHeC.lhe \
  --lhec-bkg     /home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHeC_SM/Events/run_01/singletop_FCNC_LHeC_SM.lhe \
  --lhmuc-signal /home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHmuC/Events/run_01/singletop_FCNC_LHmuC.lhe \
  --lhmuc-bkg    /home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHmuC_SM/Events/run_01/singletop_FCNC_LHmuC_SM.lhe \
  --kappa0 1e-3 \
  --observable lep \
  --luminosity 1000 \
  --deltaB 0.10 \
  --method hybrid \
  --out-prefix plots_v4


"""


# compare_LHeC_LHmuC_tails_v3.py
#
# Tail-cut comparison of FCNC single-top signal vs SM background at LHeC and LHμC.
# Outputs:
#  - dσ/dpT plots (signal vs bkg) for each collider
#  - normalized signal-shape comparison
#  - cumulative tail σ(pT>X)
#  - Z vs cut
#  - κ95 vs cut + improvement ratio κ95(LHμC)/κ95(LHeC)
#
# Notes:
#  - Event-level parsing keeps ALL events; missing objects are NaN and fail cuts.
#  - dσ/dpT is normalized as: σ_tot * (N_bin/N_tot) / ΔpT.
#  - Signal is assumed to scale as κ^2; samples are generated at κ = kappa0.
#
# Example:
#   python3 compare_LHeC_LHmuC_tails_v4.py \
#     --lhec-signal  /path/LHeC_sig.lhe  --lhec-bkg  /path/LHeC_bkg.lhe \
#     --lhmuc-signal /path/LHmuC_sig.lhe --lhmuc-bkg /path/LHmuC_bkg.lhe \
#     --observable lep --luminosity 1000 --deltaB 0.1 --out-prefix run_lep

from __future__ import annotations
import argparse, math, re
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

_FLOAT = r"([0-9Ee\+\-\.]+)"

def read_head(path: str, nbytes: int = 2_000_000) -> str:
    with open(path, "rb") as f:
        return f.read(nbytes).decode("utf-8", errors="ignore")

def parse_run_info(path: str) -> dict:
    txt = read_head(path)
    out = {}
    m = re.search(r"Integrated weight \(pb\)\s*:\s*" + _FLOAT, txt)
    if m: out["xsec_pb"] = float(m.group(1))
    for k in ["ebeam1","ebeam2","lpp1","lpp2"]:
        mm = re.search(r"\n\s*" + _FLOAT + r"\s*=\s*" + k + r"\b", txt)
        if mm: out[k] = float(mm.group(1))
    return out

def sqrt_s_approx(ebeam1: float, ebeam2: float) -> float:
    return math.sqrt(max(0.0, 4.0*ebeam1*ebeam2))  # GeV

def pt(px: float, py: float) -> float:
    return math.hypot(px, py)

@dataclass
class Sample:
    name: str
    lhe: str
    xsec_pb: float
    ebeam1: Optional[float]
    ebeam2: Optional[float]
    n: int = 0
    lep_pt: Optional[np.ndarray] = None  # event-level, NaN if not found
    b_pt: Optional[np.ndarray] = None    # event-level, NaN if not found

def parse_lhe_pts(lhe: str, max_events: Optional[int]=None, smear_rel: float=0.0, seed: int=12345
                  ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Per-event:
      lep_pt = hardest final-state e+ or μ+ (pdg -11,-13), NaN if absent
      b_pt   = hardest final-state b/bbar (pdg ±5), NaN if absent
    """
    rng = np.random.default_rng(seed)
    POS_LEP = {-11,-13}
    B = {5,-5}

    lep_pts: List[float] = []
    b_pts: List[float] = []

    in_event = False
    cur_leps, cur_bs = [], []
    n = 0

    def finalize():
        if cur_leps: lpt = max(cur_leps)
        else:        lpt = float("nan")
        if cur_bs:   bpt = max(cur_bs)
        else:        bpt = float("nan")
        if smear_rel>0:
            if not math.isnan(lpt): lpt = max(0.0, lpt*(1.0+rng.normal(0.0, smear_rel)))
            if not math.isnan(bpt): bpt = max(0.0, bpt*(1.0+rng.normal(0.0, smear_rel)))
        lep_pts.append(lpt); b_pts.append(bpt)

    with open(lhe, "r", errors="ignore") as f:
        for line in f:
            if "<event" in line:
                in_event = True
                cur_leps, cur_bs = [], []
                continue
            if in_event and "</event" in line:
                finalize()
                n += 1
                in_event = False
                if max_events is not None and n >= max_events: break
                continue
            if not in_event: continue

            parts = line.strip().split()
            if len(parts) < 11: continue
            try:
                pdg = int(parts[0]); status = int(parts[1])
                px = float(parts[6]); py = float(parts[7])
            except Exception:
                continue
            if status != 1: continue
            pT = pt(px, py)
            if pdg in POS_LEP: cur_leps.append(pT)
            if pdg in B:       cur_bs.append(pT)

    return n, np.array(lep_pts, float), np.array(b_pts, float)

def tail_count(v: np.ndarray, cut: float) -> int:
    return int(np.sum(np.isfinite(v) & (v > cut)))

def tail_xsec(sigma_pb: float, n_pass: int, n_tot: int, zero_mode: str) -> float:
    if n_tot <= 0: return 0.0
    if n_pass > 0: return sigma_pb * (n_pass/n_tot)
    if zero_mode == "zero": return 0.0
    if zero_mode == "half": return sigma_pb * (0.5/n_tot)
    if zero_mode == "cl95": return sigma_pb * (3.0/n_tot)
    raise ValueError("zero_mode must be zero|half|cl95")

def dsig_dpT(v: np.ndarray, sigma_pb: float, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_tot = len(v)
    vv = v[np.isfinite(v)]
    counts, edges = np.histogram(vv, bins=bins)
    widths = np.diff(edges)
    centers = 0.5*(edges[:-1]+edges[1:])
    if n_tot<=0: return centers, np.zeros_like(centers)
    return centers, sigma_pb*(counts/n_tot)/np.where(widths>0,widths,1.0)

def norm_shape(v: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vv = v[np.isfinite(v)]
    counts, edges = np.histogram(vv, bins=bins)
    widths = np.diff(edges)
    centers = 0.5*(edges[:-1]+edges[1:])
    area = counts.sum()
    if area<=0: return centers, np.zeros_like(centers)
    return centers, (counts/area)/np.where(widths>0,widths,1.0)

def poisson_cdf(n: int, mean: float) -> float:
    term = math.exp(-mean); s = term
    for k in range(1, n+1):
        term *= mean/k
        s += term
    return max(0.0, min(1.0, s))

def s95_poisson_asimov(b: float, cl: float=0.95) -> float:
    n = int(round(b))
    target = 1.0 - cl
    lo, hi = 0.0, max(10.0, 10.0*math.sqrt(b+1.0)+10.0)
    for _ in range(80):
        mid = 0.5*(lo+hi)
        if poisson_cdf(n, mid+b) > target: lo = mid
        else: hi = mid
    return hi

def kappa95(S0: float, B: float, kappa0: float, deltaB: float, method: str, alpha95: float) -> float:
    if S0 <= 0: return float("inf")
    if method == "gauss":
        s95 = alpha95*math.sqrt(max(0.0, B + (deltaB*B)**2))
    elif method == "poisson":
        s95 = s95_poisson_asimov(B, 0.95) if B <= 100 else alpha95*math.sqrt(B)
    else:  # hybrid
        if B < 50:
            s95 = s95_poisson_asimov(B, 0.95)
        else:
            s95 = alpha95*math.sqrt(max(0.0, B + (deltaB*B)**2))
    mu95 = s95/S0
    return kappa0*math.sqrt(mu95) if mu95>0 else float("inf")

def Z_simple(S: float, B: float, deltaB: float) -> float:
    if B<=0: return 0.0
    return S/math.sqrt(B + (deltaB*B)**2)

def savefig(path: str):
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

def build_default_cuts(max_pt: float) -> np.ndarray:
    base = np.array([50,100,150,200,250,300,350,400,500,600,800,1000], float)
    if not np.isfinite(max_pt) or max_pt<=0: return base[:6]
    return base[base < 0.95*max_pt] if (base < 0.95*max_pt).any() else base[:6]

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--lhec-signal", required=True)
    ap.add_argument("--lhec-bkg", required=True)
    ap.add_argument("--lhmuc-signal", required=True)
    ap.add_argument("--lhmuc-bkg", required=True)
    ap.add_argument("--lhec-banner-signal", default=None)
    ap.add_argument("--lhec-banner-bkg", default=None)
    ap.add_argument("--lhmuc-banner-signal", default=None)
    ap.add_argument("--lhmuc-banner-bkg", default=None)

    ap.add_argument("--observable", choices=["lep","b"], default="lep")
    ap.add_argument("--kappa0", type=float, default=1e-3)
    ap.add_argument("--luminosity", type=float, default=1000.0)  # fb^-1
    ap.add_argument("--deltaB", type=float, default=0.0)
    ap.add_argument("--method", choices=["gauss","poisson","hybrid"], default="hybrid")
    ap.add_argument("--alpha95", type=float, default=1.64)
    ap.add_argument("--mc-zero-mode", choices=["zero","half","cl95"], default="cl95")
    ap.add_argument("--smear-rel", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--cuts", default="auto")
    ap.add_argument("--nbins", type=int, default=60)
    ap.add_argument("--out-prefix", default="cmp")
    args = ap.parse_args()

    def load(name: str, lhe: str, banner: Optional[str]) -> Sample:
        info = parse_run_info(banner if banner else lhe)
        if "xsec_pb" not in info: raise RuntimeError(f"Cannot parse xsec from {banner or lhe}")
        return Sample(name, lhe, info["xsec_pb"], info.get("ebeam1"), info.get("ebeam2"))

    lhec_sig = load("LHeC@1.2 TeV", args.lhec_signal, args.lhec_banner_signal)
    lhec_bkg = load("LHeC@1.2 TeV", args.lhec_bkg,   args.lhec_banner_bkg)
    lhmuc_sig = load("LHμC@3.7 TeV", args.lhmuc_signal, args.lhmuc_banner_signal)
    lhmuc_bkg = load("LHμC@3.7 TeV", args.lhmuc_bkg,   args.lhmuc_banner_bkg)

    for s in [lhec_sig, lhec_bkg, lhmuc_sig, lhmuc_bkg]:
        s.n, s.lep_pt, s.b_pt = parse_lhe_pts(s.lhe, args.max_events, args.smear_rel, args.seed)

    def info_line(s: Sample) -> str:
        if s.ebeam1 and s.ebeam2:
            return f"{s.name}: ebeam1={s.ebeam1:.1f} GeV, ebeam2={s.ebeam2:.1f} GeV, sqrt(s)≈{sqrt_s_approx(s.ebeam1,s.ebeam2)/1000:.3f} TeV, σ={s.xsec_pb:.6e} pb, N={s.n}"
        return f"{s.name}: σ={s.xsec_pb:.6e} pb, N={s.n}"
    print(info_line(lhec_sig)); print(info_line(lhec_bkg))
    print(info_line(lhmuc_sig)); print(info_line(lhmuc_bkg))

    def vals(s: Sample) -> np.ndarray:
        return s.lep_pt if args.observable=="lep" else s.b_pt

    maxpt = max(np.nanmax(vals(lhec_sig)), np.nanmax(vals(lhec_bkg)),
                np.nanmax(vals(lhmuc_sig)), np.nanmax(vals(lhmuc_bkg)))
    cuts = build_default_cuts(maxpt) if args.cuts.strip().lower()=="auto" else np.sort(np.array([float(x) for x in args.cuts.split(",") if x.strip()], float))
    bins = np.linspace(0.0, (maxpt if np.isfinite(maxpt) and maxpt>0 else 1000.0), args.nbins+1)

    xlabel = r"$p_T^{\ell^+}$ [GeV]" if args.observable=="lep" else r"$p_T^{b}$ [GeV]"
    which = args.observable

    # dσ/dpT: LHeC
    plt.figure(figsize=(7.2,5.2))
    x,y = dsig_dpT(vals(lhec_sig), lhec_sig.xsec_pb, bins); plt.step(x,y,where="mid",label=f"SIGNAL, σ={lhec_sig.xsec_pb:.3e} pb")
    x,y = dsig_dpT(vals(lhec_bkg), lhec_bkg.xsec_pb, bins); plt.step(x,y,where="mid",label=f"BKG, σ={lhec_bkg.xsec_pb:.3e} pb")
    plt.yscale("log"); plt.xlabel(xlabel); plt.ylabel(r"$d\sigma/dp_T$ [pb/GeV]")
    plt.title(f"LHeC@1.2 TeV\nSignal (κ={args.kappa0:g}) vs SM background\nObservable: {('decay lepton' if which=='lep' else 'b (leading b)')}")
    plt.grid(True, which="both", ls=":", alpha=0.6); plt.legend()
    savefig(f"{args.out_prefix}_LHeC_{which}_sig_vs_bkg.png")

    # dσ/dpT: LHμC
    plt.figure(figsize=(7.2,5.2))
    x,y = dsig_dpT(vals(lhmuc_sig), lhmuc_sig.xsec_pb, bins); plt.step(x,y,where="mid",label=f"SIGNAL, σ={lhmuc_sig.xsec_pb:.3e} pb")
    x,y = dsig_dpT(vals(lhmuc_bkg), lhmuc_bkg.xsec_pb, bins); plt.step(x,y,where="mid",label=f"BKG, σ={lhmuc_bkg.xsec_pb:.3e} pb")
    plt.yscale("log"); plt.xlabel(xlabel); plt.ylabel(r"$d\sigma/dp_T$ [pb/GeV]")
    plt.title(f"LHμC@3.7 TeV\nSignal (κ={args.kappa0:g}) vs SM background\nObservable: {('decay lepton' if which=='lep' else 'b (leading b)')}")
    plt.grid(True, which="both", ls=":", alpha=0.6); plt.legend()
    savefig(f"{args.out_prefix}_LHmuC_{which}_sig_vs_bkg.png")

    # normalized signal shape
    plt.figure(figsize=(7.2,5.2))
    x,y = norm_shape(vals(lhec_sig), bins); plt.step(x,y,where="mid",label="LHeC@1.2 TeV (signal)")
    x,y = norm_shape(vals(lhmuc_sig), bins); plt.step(x,y,where="mid",label="LHμC@3.7 TeV (signal)")
    plt.yscale("log"); plt.xlabel(xlabel); plt.ylabel(r"Normalized $(1/\sigma)\,d\sigma/dp_T$ [1/GeV]")
    plt.title(f"Signal shape (normalized)\n{('Decay lepton' if which=='lep' else 'b-jet')} $p_T$: LHeC vs LHμC")
    plt.grid(True, which="both", ls=":", alpha=0.6); plt.legend()
    savefig(f"{args.out_prefix}_signal_shape_LHeC_vs_LHmuC_{which}.png")

    # tail scan
    L_pb = args.luminosity*1e3  # fb^-1 -> pb^-1
    def scan(sample_sig: Sample, sample_bkg: Sample) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        sig_tail=[]; bkg_tail=[]; Zs=[]; ks=[]
        for c in cuts:
            nS = tail_count(vals(sample_sig), c)
            nB = tail_count(vals(sample_bkg), c)
            sigpb = tail_xsec(sample_sig.xsec_pb, nS, sample_sig.n, args.mc_zero_mode)
            bkgpb = tail_xsec(sample_bkg.xsec_pb, nB, sample_bkg.n, args.mc_zero_mode)
            sig_tail.append(sigpb); bkg_tail.append(bkgpb)
            S0 = L_pb*sigpb; B = L_pb*bkgpb
            Zs.append(Z_simple(S0, B, args.deltaB))
            ks.append(kappa95(S0, B, args.kappa0, args.deltaB, args.method, args.alpha95))
        return np.array(sig_tail), np.array(bkg_tail), np.array(Zs), np.array(ks)

    S1,B1,Z1,K1 = scan(lhec_sig, lhec_bkg)
    S2,B2,Z2,K2 = scan(lhmuc_sig, lhmuc_bkg)
    ratio = np.divide(K2, K1, out=np.full_like(K2, np.nan), where=np.isfinite(K1)&(K1>0))

    # cumulative tail σ
    plt.figure(figsize=(7.2,5.2))
    plt.plot(cuts,S1,marker="o",label="LHeC signal")
    plt.plot(cuts,B1,marker="o",ls="--",label="LHeC bkg")
    plt.plot(cuts,S2,marker="o",label="LHμC signal")
    plt.plot(cuts,B2,marker="o",ls="--",label="LHμC bkg")
    plt.yscale("log"); plt.xlabel(f"Tail cut: {xlabel} > X [GeV]")
    plt.ylabel(r"Tail cross section $\sigma(p_T>X)$ [pb]")
    plt.title("Cumulative tail cross sections (monotonic)\nSolid=signal, dashed=background")
    plt.grid(True, which="both", ls=":", alpha=0.6); plt.legend()
    savefig(f"{args.out_prefix}_tail_xsec_{which}_LHeC_vs_LHmuC.png")

    # Z vs cut
    plt.figure(figsize=(7.2,5.2))
    plt.plot(cuts,Z1,marker="o",label="LHeC@1.2 TeV")
    plt.plot(cuts,Z2,marker="o",label="LHμC@3.7 TeV")
    plt.xlabel(f"Tail cut: {xlabel} > X [GeV]"); plt.ylabel("Approx. significance Z")
    plt.title(f"Approx. significance vs tail cut\nL={args.luminosity:g} fb$^{{-1}}$, δB={args.deltaB:.2f}")
    plt.grid(True, which="both", ls=":", alpha=0.6); plt.legend()
    savefig(f"{args.out_prefix}_Z_vs_tailcut_{which}.png")

    # kappa95 vs cut
    plt.figure(figsize=(7.2,5.2))
    plt.plot(cuts,K1,marker="o",label="LHeC@1.2 TeV")
    plt.plot(cuts,K2,marker="o",label="LHμC@3.7 TeV")
    plt.yscale("log"); plt.xlabel(f"Tail cut: {xlabel} > X [GeV]")
    plt.ylabel(r"Expected limit on coupling $\kappa_{95}$")
    plt.title(f"Expected coupling reach vs tail cut (using {xlabel})\nL={args.luminosity:g} fb$^{{-1}}$, δB={args.deltaB:.2f}, method={args.method}")
    plt.grid(True, which="both", ls=":", alpha=0.6); plt.legend()
    savefig(f"{args.out_prefix}_kappa95_vs_tailcut_{which}.png")

    # improvement ratio
    plt.figure(figsize=(7.2,5.2))
    plt.plot(cuts,ratio,marker="o"); plt.axhline(1.0,ls="--")
    plt.yscale("log"); plt.xlabel(f"Tail cut: {xlabel} > X [GeV]")
    plt.ylabel(r"$\kappa_{95}(\mathrm{LH}\mu\mathrm{C})/\kappa_{95}(\mathrm{LHeC})$")
    plt.title("Improvement factor (<1 means LHμC gives stronger limit)")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    savefig(f"{args.out_prefix}_improvement_ratio_kappa95_LHmuC_over_LHeC_{which}.png")

    # best points
    def best(cuts: np.ndarray, K: np.ndarray) -> Tuple[float,float]:
        m = np.isfinite(K)
        if not m.any(): return float("nan"), float("inf")
        i = np.argmin(K[m])
        return float(cuts[m][i]), float(K[m][i])

    c1,k1 = best(cuts,K1); c2,k2 = best(cuts,K2)
    print("\n=== Summary ===")
    print(f"Observable: {('decay lepton pT' if which=='lep' else 'b pT')}")
    print(f"L={args.luminosity:g} fb^-1, δB={args.deltaB:.2f}, method={args.method}, zero_mode={args.mc_zero_mode}")
    print(f"Best LHeC:  X={c1:.1f} GeV -> κ95={k1:.3e}")
    print(f"Best LHμC:  X={c2:.1f} GeV -> κ95={k2:.3e}")
    if math.isfinite(k1) and k1>0 and math.isfinite(k2):
        print(f"Ratio (LHμC/LHeC) at best points: {k2/k1:.3f} (<1 means LHμC stronger)")

    print("\nSaved PNGs:")
    for fn in [
        f"{args.out_prefix}_LHeC_{which}_sig_vs_bkg.png",
        f"{args.out_prefix}_LHmuC_{which}_sig_vs_bkg.png",
        f"{args.out_prefix}_signal_shape_LHeC_vs_LHmuC_{which}.png",
        f"{args.out_prefix}_tail_xsec_{which}_LHeC_vs_LHmuC.png",
        f"{args.out_prefix}_Z_vs_tailcut_{which}.png",
        f"{args.out_prefix}_kappa95_vs_tailcut_{which}.png",
        f"{args.out_prefix}_improvement_ratio_kappa95_LHmuC_over_LHeC_{which}.png",
    ]:
        print("  *", fn)

if __name__ == "__main__":
    main()

