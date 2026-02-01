#!/usr/bin/env python3


"""

python3 compare_LHeC_LHmuC_tails_v3.py \
  --lhec-sig  /home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHeC/Events/run_01/singletop_FCNC_LHeC.lhe \
  --lhec-bkg  /home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHeC_SM/Events/run_01/singletop_FCNC_LHeC_SM.lhe \
  --lhmuc-sig /home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHmuC/Events/run_01/singletop_FCNC_LHmuC.lhe \
  --lhmuc-bkg /home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHmuC_SM/Events/run_01/singletop_FCNC_LHmuC_SM.lhe \
  --kappa0 1e-3 \
  --outdir plots_v3

"""

"""
compare_LHeC_LHmuC_tails_v3.py

Truth-level LHE analysis for FCNC single-top (signal) vs SM background at LHeC and LHμC.

What this script is designed to show (cleanly):
1) The LHμC signal tends to populate harder pT tails (boosted kinematics at higher √s).
2) In the high-pT tail region, the *background falls faster* than the signal → better S/B and better κ95.
3) Using the same "tail cut" strategy (and optionally an optimized 2D cut), LHμC can yield a *stronger* expected limit.

Key features:
- Robust LHE (v2/v3) parsing (plain or .gz)
- Reads beam energies (ebeam1/2, lpp1/lpp2) + total cross section from <init> or MGGenerationInfo
- Observables:
    * decay lepton pT, eta  (l+ from W in top decay)
    * b-jet pT if present, otherwise leading-jet pT
- LHμC (μ⁺p) ambiguity: two μ⁺ candidates (scattered + decay)
    * selects decay μ⁺ by tracing ancestry to W (PDG=24)
    * reports naive mis-tag rate for "highest pT μ⁺" strategy
- Plots:
    * dσ/dpT (signal vs bkg) for lepton and jet (each collider)
    * normalized signal shape: LHeC vs LHμC (lepton pT)
    * cumulative tail cross sections σ(pT>X)
    * tail S/B and tail significance (at κ0) vs cut
    * expected coupling reach κ95 vs tail cut
    * improvement ratio κ95(LHμC)/κ95(LHeC)
    * 2D cut optimization (pTlep, pTjet) heatmaps + best κ95 vs luminosity

Coupling scaling:
- Assumes signal yield ∝ κ^2.
- Your signal LHE is generated at κ = κ0 (default 1e-3), so:
      σ_sig(κ) = σ_sig(κ0) * (κ/κ0)^2

Expected 95% CL limit:
- Default cut-and-count approximation:
      if B ~ 0:  s95 ≈ 3.0 events
      else:      s95 = 1.64 * sqrt(B + (δB*B)^2)
  and κ95 = κ0 * sqrt(s95 / S(κ0)).

Usage example:
  python3 compare_LHeC_LHmuC_tails_v3.py \
    --lhec-sig  LHeC_signal.lhe.gz  --lhec-bkg  LHeC_bkg.lhe.gz \
    --lhmuc-sig LHmuC_signal.lhe.gz --lhmuc-bkg LHmuC_bkg.lhe.gz \
    --kappa0 1e-3 --lumi 1000 --deltaB 0.0 --outdir plots_v3

"""

from __future__ import annotations
import argparse
import gzip
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Generic helpers
# -----------------------------

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")

def _pt(px: float, py: float) -> float:
    return float(math.hypot(px, py))

def _eta(px: float, py: float, pz: float) -> float:
    p = math.sqrt(px*px + py*py + pz*pz)
    # protect against division by zero / exact collinear
    if p <= abs(pz):
        return float(np.sign(pz) * 1e6)
    return float(0.5 * math.log((p + pz) / (p - pz)))

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _as_list_of_floats(s: str) -> List[float]:
    # accepts "50,100,150" or "50 100 150"
    parts = re.split(r"[,\s]+", s.strip())
    return [float(p) for p in parts if p]

def _sqrt_s_ep(ebeam_p: float, ebeam_l: float) -> float:
    # ultra-relativistic approximation: sqrt(s) ≈ 2*sqrt(Ep*El)
    return 2.0 * math.sqrt(ebeam_p * ebeam_l) / 1000.0  # TeV


# -----------------------------
# LHE parsing
# -----------------------------

@dataclass
class BeamInfo:
    lpp1: int
    lpp2: int
    ebeam1: float
    ebeam2: float

@dataclass
class Sample:
    label: str
    path: str
    sigma_pb: float
    beam: BeamInfo
    n_events: int

    lep_pt: np.ndarray
    lep_eta: np.ndarray
    jet_pt: np.ndarray

    # LHμC-only diagnostics (empty for LHeC)
    mu_scatter_pt: np.ndarray
    mu_decay_pt: np.ndarray
    mu_decay_is_highest_pt: Optional[float]  # fraction (if computed)

@dataclass
class Particle:
    pdg: int
    status: int
    moth1: int
    moth2: int
    px: float
    py: float
    pz: float
    E: float

def _trace_has_ancestor(pidx: int, parts: List[Particle], target_pdgs: Set[int], max_depth: int = 30) -> bool:
    """
    Return True if particle pidx has an ancestor with PDG in target_pdgs.
    pidx is 0-based; LHE mothers are 1-based.
    """
    seen = set()
    stack = [pidx]
    depth = 0
    while stack and depth < max_depth:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        p = parts[cur]
        m1 = p.moth1 - 1
        m2 = p.moth2 - 1
        for m in (m1, m2):
            if m < 0 or m >= len(parts):
                continue
            mp = parts[m]
            if mp.pdg in target_pdgs:
                return True
            stack.append(m)
        depth += 1
    return False

def _read_sigma_from_init_lines(lines: List[str]) -> Optional[float]:
    """
    <init> block usually contains:
      line0: IDs/ebeams/...
      line1: xsec xerr xmax ...   (first float is xsec in pb)
    """
    if len(lines) < 2:
        return None
    floats = _FLOAT_RE.findall(lines[1])
    if not floats:
        return None
    try:
        return float(floats[0])
    except Exception:
        return None

def _read_beam_from_init_line(line: str) -> Optional[BeamInfo]:
    # expected: IDBMUP1 IDBMUP2 EBMUP1 EBMUP2 ...
    toks = line.strip().split()
    if len(toks) < 4:
        return None
    try:
        ebeam1 = float(toks[2])
        ebeam2 = float(toks[3])
        return BeamInfo(lpp1=0, lpp2=0, ebeam1=ebeam1, ebeam2=ebeam2)
    except Exception:
        return None

def _read_beam_lpp_from_header(header_text: str) -> Tuple[Optional[int], Optional[int]]:
    m1 = re.search(r"^\s*([-\d]+)\s*=\s*lpp1\b", header_text, flags=re.MULTILINE)
    m2 = re.search(r"^\s*([-\d]+)\s*=\s*lpp2\b", header_text, flags=re.MULTILINE)
    lpp1 = int(m1.group(1)) if m1 else None
    lpp2 = int(m2.group(1)) if m2 else None
    return lpp1, lpp2

def _read_sigma_from_mggeninfo(header_text: str) -> Optional[float]:
    m = re.search(r"Integrated weight\s*\(pb\)\s*:\s*([-\d.eE+]+)", header_text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _read_ebeam_from_header(header_text: str) -> Optional[Tuple[float, float]]:
    mE1 = re.search(r"^\s*([-\d.eE+]+)\s*=\s*ebeam1\b", header_text, flags=re.MULTILINE)
    mE2 = re.search(r"^\s*([-\d.eE+]+)\s*=\s*ebeam2\b", header_text, flags=re.MULTILINE)
    if not (mE1 and mE2):
        return None
    return float(mE1.group(1)), float(mE2.group(1))

def load_lhe_sample(
    path: str,
    label: str,
    collider: str,
    max_events: Optional[int] = None,
    prefer_decay_from_W: bool = True,
) -> Sample:
    """
    collider: "LHeC" or "LHmuC" (controls μ⁺ ambiguity handling)
    """

    lep_pts: List[float] = []
    lep_etas: List[float] = []
    jet_pts: List[float] = []

    mu_scatter_pts: List[float] = []
    mu_decay_pts: List[float] = []
    mu_decay_is_highest_flags: List[int] = []

    header_chunks: List[str] = []
    init_lines: List[str] = []

    in_header = False
    in_init = False
    in_event = False

    n_events = 0
    sigma_pb: Optional[float] = None
    beam: Optional[BeamInfo] = None

    with _open_text(path) as f:
        for raw in f:
            line = raw.strip()

            # header block
            if line.startswith("<header"):
                in_header = True
            if in_header:
                header_chunks.append(raw)
            if line.startswith("</header"):
                in_header = False

            # init block
            if line.startswith("<init"):
                in_init = True
                init_lines = []
                continue
            if in_init:
                if line.startswith("</init"):
                    in_init = False
                    if init_lines:
                        b = _read_beam_from_init_line(init_lines[0])
                        if b:
                            beam = b
                        s = _read_sigma_from_init_lines(init_lines)
                        if s is not None:
                            sigma_pb = s
                    continue
                init_lines.append(raw)
                continue

            # event block start
            if line.startswith("<event"):
                in_event = True
                event_lines: List[str] = []
                continue

            if in_event:
                if line.startswith("</event"):
                    in_event = False

                    if not event_lines:
                        continue

                    head = event_lines[0].split()
                    if len(head) < 1:
                        continue
                    try:
                        nup = int(head[0])
                    except Exception:
                        continue

                    particle_lines = event_lines[1:1+nup]
                    parts: List[Particle] = []
                    for pl in particle_lines:
                        toks = pl.split()
                        if len(toks) < 10:
                            continue
                        pdg = int(toks[0])
                        status = int(toks[1])
                        moth1 = int(toks[2])
                        moth2 = int(toks[3])
                        px = float(toks[6]); py = float(toks[7]); pz = float(toks[8]); E = float(toks[9])
                        parts.append(Particle(pdg=pdg, status=status, moth1=moth1, moth2=moth2, px=px, py=py, pz=pz, E=E))

                    # build candidates
                    jet_candidates = []   # (idx, pt)
                    b_candidates = []     # (idx, pt)
                    lplus_candidates = [] # e+, mu+ (pdg -11, -13)
                    muplus_candidates = []# mu+ (pdg -13), includes scatter+decay at LHμC

                    for i, p in enumerate(parts):
                        if p.status != 1:
                            continue
                        apdg = abs(p.pdg)
                        if p.pdg == 21 or (1 <= apdg <= 5):
                            jet_candidates.append((i, _pt(p.px, p.py)))
                        if p.pdg == 5:
                            b_candidates.append((i, _pt(p.px, p.py)))
                        if p.pdg in (-11, -13):
                            lplus_candidates.append((i, _pt(p.px, p.py)))
                        if p.pdg == -13:
                            muplus_candidates.append((i, _pt(p.px, p.py)))

                    # jet observable: b-pt if present else leading jet
                    leadjet_pt = max((pt for _, pt in jet_candidates), default=float("nan"))
                    bpt = max((pt for _, pt in b_candidates), default=float("nan"))
                    jet_pt_obs = bpt if np.isfinite(bpt) else leadjet_pt

                    # decay lepton observable
                    lep_pt_obs = float("nan")
                    lep_eta_obs = float("nan")

                    if collider.lower() == "lhec":
                        # scattered lepton is e- (pdg 11). decay lepton is l+ (pdg -11/-13).
                        if lplus_candidates:
                            idx = max(lplus_candidates, key=lambda x: x[1])[0]
                            pp = parts[idx]
                            lep_pt_obs = _pt(pp.px, pp.py)
                            lep_eta_obs = _eta(pp.px, pp.py, pp.pz)

                    elif collider.lower() == "lhmuc":
                        # beam is mu+ => there can be 2 mu+ (scatter + decay)
                        decay_idx = None
                        scatter_idx = None

                        if muplus_candidates:
                            if prefer_decay_from_W:
                                for idx, _ in muplus_candidates:
                                    if _trace_has_ancestor(idx, parts, target_pdgs={24}):
                                        decay_idx = idx
                                        break
                            if decay_idx is None:
                                # fallback: choose highest pT mu+ as decay
                                decay_idx = max(muplus_candidates, key=lambda x: x[1])[0]

                            others = [idx for idx, _ in muplus_candidates if idx != decay_idx]
                            if others:
                                scatter_idx = max(others, key=lambda j: _pt(parts[j].px, parts[j].py))

                            if scatter_idx is not None:
                                mu_scatter_pts.append(_pt(parts[scatter_idx].px, parts[scatter_idx].py))
                            if decay_idx is not None:
                                mu_decay_pts.append(_pt(parts[decay_idx].px, parts[decay_idx].py))
                                highest_idx = max(muplus_candidates, key=lambda x: x[1])[0]
                                mu_decay_is_highest_flags.append(1 if decay_idx == highest_idx else 0)

                                pp = parts[decay_idx]
                                lep_pt_obs = _pt(pp.px, pp.py)
                                lep_eta_obs = _eta(pp.px, pp.py, pp.pz)

                    # commit observables (keep NaNs if selection failed; later we warn if this happens)
                    lep_pts.append(float(lep_pt_obs))
                    lep_etas.append(float(lep_eta_obs))
                    jet_pts.append(float(jet_pt_obs))

                    n_events += 1
                    if max_events is not None and n_events >= max_events:
                        break
                    continue

                # still inside event
                if line and not line.startswith("#"):
                    event_lines.append(line)
                continue

            if max_events is not None and n_events >= max_events:
                break

    header_text = "".join(header_chunks)

    # lpp1/lpp2 from header
    lpp1, lpp2 = _read_beam_lpp_from_header(header_text)

    # sigma from MGGenerationInfo if needed
    if sigma_pb is None:
        sigma_pb = _read_sigma_from_mggeninfo(header_text)

    if sigma_pb is None:
        raise RuntimeError(f"Could not determine total cross section (pb) from: {path}")

    # beam energies
    if beam is None:
        ebeams = _read_ebeam_from_header(header_text)
        if ebeams is None:
            raise RuntimeError(f"Could not determine beam energies from: {path}")
        beam = BeamInfo(lpp1=(lpp1 if lpp1 is not None else 0),
                        lpp2=(lpp2 if lpp2 is not None else 0),
                        ebeam1=float(ebeams[0]),
                        ebeam2=float(ebeams[1]))
    else:
        beam = BeamInfo(lpp1=(lpp1 if lpp1 is not None else beam.lpp1),
                        lpp2=(lpp2 if lpp2 is not None else beam.lpp2),
                        ebeam1=beam.ebeam1,
                        ebeam2=beam.ebeam2)

    lep_pt_arr = np.array(lep_pts, dtype=float)
    lep_eta_arr = np.array(lep_etas, dtype=float)
    jet_pt_arr = np.array(jet_pts, dtype=float)

    mu_scatter_arr = np.array(mu_scatter_pts, dtype=float)
    mu_decay_arr = np.array(mu_decay_pts, dtype=float)
    mu_decay_is_highest = float(np.mean(mu_decay_is_highest_flags)) if mu_decay_is_highest_flags else None

    return Sample(
        label=label,
        path=path,
        sigma_pb=float(sigma_pb),
        beam=beam,
        n_events=int(n_events),
        lep_pt=lep_pt_arr,
        lep_eta=lep_eta_arr,
        jet_pt=jet_pt_arr,
        mu_scatter_pt=mu_scatter_arr,
        mu_decay_pt=mu_decay_arr,
        mu_decay_is_highest_pt=mu_decay_is_highest,
    )


# -----------------------------
# Physics utilities
# -----------------------------

def build_default_bins(max_x: float, step: float) -> np.ndarray:
    max_x = float(max_x)
    step = float(step)
    if not np.isfinite(max_x) or max_x <= 0:
        max_x = 100.0
    nb = int(math.ceil(max_x / step))
    return np.linspace(0.0, nb * step, nb + 1)

def dsigma_hist(values: np.ndarray, sigma_pb: float, n_events: int, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    dσ/dx [pb/unit] for unweighted events.
    Uses weight per event = sigma_pb / n_events.
    NaNs are ignored (treated as "failed reconstruction"); we warn if many exist.
    """
    v = values[np.isfinite(values)]
    counts, _ = np.histogram(v, bins=bins)
    widths = np.diff(bins)
    w_evt = sigma_pb / float(n_events)
    ds = counts.astype(float) * w_evt / widths
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, ds

def normalized_shape(values: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalized density: (1/σ) dσ/dx [1/unit]
    """
    v = values[np.isfinite(values)]
    counts, _ = np.histogram(v, bins=bins)
    widths = np.diff(bins)
    area = float(np.sum(counts * widths))
    centers = 0.5 * (bins[:-1] + bins[1:])
    if area <= 0:
        return centers, np.zeros_like(centers)
    dens = counts.astype(float) / area
    return centers, dens

def tail_sigma(values: np.ndarray, sigma_pb: float, cuts: np.ndarray) -> np.ndarray:
    """
    σ(x > cut) in pb for each cut.
    Denominator is total event count (including NaNs → treated as not passing).
    """
    n_tot = values.size
    if n_tot == 0:
        return np.zeros_like(cuts, dtype=float)

    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.zeros_like(cuts, dtype=float)

    v_sorted = np.sort(v)
    idx = np.searchsorted(v_sorted, cuts, side="right")
    n_pass = (v_sorted.size - idx).astype(float)
    frac = n_pass / float(n_tot)
    return sigma_pb * frac

def expected_sb_and_significance(
    sig_tail_pb: np.ndarray,
    bkg_tail_pb: np.ndarray,
    lumi_fb: float,
    kappa0: float,
    deltaB: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each cut, compute:
      S/B at κ0
      Z = S / sqrt(B + (δB B)^2) at κ0  (Gaussian proxy)
    """
    L_pb = lumi_fb * 1e3
    S = sig_tail_pb * L_pb
    B = bkg_tail_pb * L_pb

    with np.errstate(divide="ignore", invalid="ignore"):
        sb = S / B
        denom = np.sqrt(B + (deltaB * B)**2)
        Z = np.where(denom > 0, S / denom, np.nan)

    return sb, Z

def kappa95_cutcount(
    sig_tail_pb: np.ndarray,
    bkg_tail_pb: np.ndarray,
    lumi_fb: float,
    kappa0: float,
    deltaB: float = 0.0
) -> np.ndarray:
    """
    Expected 95% CL κ limit using cut-and-count in tail.
    """
    L_pb = lumi_fb * 1e3
    S0 = sig_tail_pb * L_pb
    B = bkg_tail_pb * L_pb

    out = np.full_like(S0, np.nan, dtype=float)
    for i, (s0, b) in enumerate(zip(S0, B)):
        if s0 <= 0:
            continue
        if b < 1e-9:
            s95 = 3.0
        else:
            s95 = 1.64 * math.sqrt(b + (deltaB*b)**2)
        out[i] = kappa0 * math.sqrt(s95 / s0)
    return out


# -----------------------------
# Plotting
# -----------------------------

def _finish_plot(outdir: str, filename: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=200)
    plt.close()

def plot_dsigma(outdir: str, title: str, xlabel: str,
                centers: np.ndarray, ds_sig: np.ndarray, ds_bkg: np.ndarray,
                lab_sig: str, lab_bkg: str, filename: str) -> None:
    plt.figure()
    plt.step(centers, ds_sig, where="mid", label=lab_sig)
    plt.step(centers, ds_bkg, where="mid", label=lab_bkg)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(r"$d\sigma/dx$ [pb/GeV]")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    _finish_plot(outdir, filename)

def plot_shape(outdir: str, title: str, xlabel: str,
               centers: np.ndarray, dens_a: np.ndarray, dens_b: np.ndarray,
               lab_a: str, lab_b: str, filename: str) -> None:
    plt.figure()
    plt.step(centers, dens_a, where="mid", label=lab_a)
    plt.step(centers, dens_b, where="mid", label=lab_b)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(r"Normalized $(1/\sigma)\,d\sigma/dx$ [1/GeV]")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    _finish_plot(outdir, filename)

def plot_tail_xsec(outdir: str, cuts: np.ndarray,
                   sig_L: np.ndarray, bkg_L: np.ndarray,
                   sig_M: np.ndarray, bkg_M: np.ndarray,
                   filename: str) -> None:
    plt.figure()
    plt.plot(cuts, sig_L, marker="o", label="LHeC signal")
    plt.plot(cuts, bkg_L, marker="o", linestyle="--", label="LHeC bkg")
    plt.plot(cuts, sig_M, marker="o", label=r"LH$\mu$C signal")
    plt.plot(cuts, bkg_M, marker="o", linestyle="--", label=r"LH$\mu$C bkg")
    plt.title("Cumulative tail cross sections (monotonic)\nSolid=signal, dashed=background")
    plt.xlabel(r"Tail cut: $p_T^{\ell^+} > X$ [GeV]")
    plt.ylabel(r"$\sigma(p_T^{\ell^+} > X)$ [pb]")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    _finish_plot(outdir, filename)

def plot_kappa_vs_cut(outdir: str, cuts: np.ndarray,
                      k_L: np.ndarray, k_M: np.ndarray,
                      lumi_fb: float, deltaB: float,
                      filename: str) -> None:
    plt.figure()
    plt.plot(cuts, k_L, marker="o", label="LHeC")
    plt.plot(cuts, k_M, marker="o", label=r"LH$\mu$C")
    plt.title(rf"Expected coupling reach vs tail cut (using $p_T^{{\ell^+}}$)"
              + "\n" + rf"$L$={lumi_fb:.0f} fb$^{{-1}}$, $\delta_B$={deltaB:.2f}")
    plt.xlabel(r"Tail cut: $p_T^{\ell^+} > X$ [GeV]")
    plt.ylabel(r"Expected limit on coupling $\kappa_{95}$")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    _finish_plot(outdir, filename)

def plot_improvement_ratio(outdir: str, cuts: np.ndarray, ratio: np.ndarray, filename: str) -> None:
    plt.figure()
    plt.plot(cuts, ratio, marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.title("Improvement factor (<1 means LHμC gives stronger limit)")
    plt.xlabel(r"Tail cut: $p_T^{\ell^+} > X$ [GeV]")
    plt.ylabel(r"$\kappa_{95}(\mathrm{LH}\mu\mathrm{C}) / \kappa_{95}(\mathrm{LHeC})$")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    _finish_plot(outdir, filename)

def plot_sb_and_Z(outdir: str, cuts: np.ndarray,
                  sb_L: np.ndarray, sb_M: np.ndarray,
                  Z_L: np.ndarray, Z_M: np.ndarray,
                  lumi_fb: float, deltaB: float) -> None:
    # S/B
    plt.figure()
    plt.plot(cuts, sb_L, marker="o", label="LHeC")
    plt.plot(cuts, sb_M, marker="o", label=r"LH$\mu$C")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.xlabel(r"Tail cut: $p_T^{\ell^+} > X$ [GeV]")
    plt.ylabel(r"$S/B$ at $\kappa_0$")
    plt.title(rf"Tail purity (S/B) at $\kappa_0$  |  $L$={lumi_fb:.0f} fb$^{{-1}}$, $\delta_B$={deltaB:.2f}")
    plt.legend()
    _finish_plot(outdir, "tail_SB_vs_cut.png")

    # significance proxy
    plt.figure()
    plt.plot(cuts, Z_L, marker="o", label="LHeC")
    plt.plot(cuts, Z_M, marker="o", label=r"LH$\mu$C")
    plt.grid(True, which="both", linestyle=":")
    plt.xlabel(r"Tail cut: $p_T^{\ell^+} > X$ [GeV]")
    plt.ylabel(r"$Z \approx S/\sqrt{B+(\delta_B B)^2}$ at $\kappa_0$")
    plt.title(rf"Tail significance proxy at $\kappa_0$  |  $L$={lumi_fb:.0f} fb$^{{-1}}$, $\delta_B$={deltaB:.2f}")
    plt.legend()
    _finish_plot(outdir, "tail_significance_vs_cut.png")

def plot_heatmap(outdir: str, title: str,
                 xcuts: np.ndarray, ycuts: np.ndarray, Z: np.ndarray,
                 xlabel: str, ylabel: str, filename: str) -> None:
    plt.figure()
    Zp = np.array(Z, dtype=float)
    Zp[~np.isfinite(Zp)] = np.nan
    with np.errstate(invalid="ignore"):
        logZ = np.log10(Zp)

    im = plt.imshow(
        logZ.T,
        origin="lower",
        aspect="auto",
        extent=[xcuts[0], xcuts[-1], ycuts[0], ycuts[-1]],
    )
    plt.colorbar(im, label=r"$\log_{10}(\kappa_{95})$")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    _finish_plot(outdir, filename)


# -----------------------------
# 2D tail utility (fast)
# -----------------------------

def tail_matrix_2d(
    x: np.ndarray,
    y: np.ndarray,
    sigma_pb: float,
    xcuts: np.ndarray,
    ycuts: np.ndarray
) -> np.ndarray:
    """
    Compute σ(x > X and y > Y) for all (X,Y) in xcuts×ycuts, efficiently.

    - Denominator is total events (including NaNs treated as failing).
    - Uses histogram2d with edges that include the thresholds.
    """
    n_tot = x.size
    if n_tot == 0:
        return np.zeros((len(xcuts), len(ycuts)), dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.zeros((len(xcuts), len(ycuts)), dtype=float)

    vx = x[mask]
    vy = y[mask]

    # edges: [0] + cuts + [max+eps]
    xmax = max(float(np.nanmax(vx) * 1.01), float(xcuts[-1]) + 1.0)
    ymax = max(float(np.nanmax(vy) * 1.01), float(ycuts[-1]) + 1.0)

    xedges = np.concatenate(([0.0], xcuts, [xmax]))
    yedges = np.concatenate(([0.0], ycuts, [ymax]))

    H, _, _ = np.histogram2d(vx, vy, bins=[xedges, yedges])

    # cumulative sum from top-right: C[i,j] = sum_{a>=i, b>=j} H[a,b]
    H_rev = H[::-1, ::-1]
    C_rev = np.cumsum(np.cumsum(H_rev, axis=0), axis=1)
    C = C_rev[::-1, ::-1]

    tail_counts = np.zeros((len(xcuts), len(ycuts)), dtype=float)
    for ix in range(len(xcuts)):
        i_bin = ix + 1  # bin index whose lower edge is xcuts[ix]
        for iy in range(len(ycuts)):
            j_bin = iy + 1
            if i_bin < C.shape[0] and j_bin < C.shape[1]:
                tail_counts[ix, iy] = C[i_bin, j_bin]

    frac = tail_counts / float(n_tot)
    return sigma_pb * frac


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lhec-sig", required=True)
    ap.add_argument("--lhec-bkg", required=True)
    ap.add_argument("--lhmuc-sig", required=True)
    ap.add_argument("--lhmuc-bkg", required=True)
    ap.add_argument("--outdir", default="plots_v3")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--kappa0", type=float, default=1e-3)
    ap.add_argument("--lumi", type=float, default=1000.0)
    ap.add_argument("--lumis", default="300,1000,3000")
    ap.add_argument("--deltaB", type=float, default=0.0)
    ap.add_argument("--tail-cuts", default="50,100,150,200,250,300,350,400")
    ap.add_argument("--jet-cuts", default="50,100,150,200,250,300,350,400")
    ap.add_argument("--bin-step-lep", type=float, default=5.0)
    ap.add_argument("--bin-step-jet", type=float, default=5.0)
    args = ap.parse_args()

    _safe_mkdir(args.outdir)

    cuts = np.sort(np.unique(np.array(_as_list_of_floats(args.tail_cuts), dtype=float)))
    jet_cuts = np.sort(np.unique(np.array(_as_list_of_floats(args.jet_cuts), dtype=float)))
    lumis = np.sort(np.unique(np.array(_as_list_of_floats(args.lumis), dtype=float)))

    # Load samples
    lhec_sig = load_lhe_sample(args.lhec_sig, "LHeC signal", "LHeC", max_events=args.max_events)
    lhec_bkg = load_lhe_sample(args.lhec_bkg, "LHeC bkg", "LHeC", max_events=args.max_events)
    lhmuc_sig = load_lhe_sample(args.lhmuc_sig, "LHmuC signal", "LHmuC", max_events=args.max_events)
    lhmuc_bkg = load_lhe_sample(args.lhmuc_bkg, "LHmuC bkg", "LHmuC", max_events=args.max_events)

    def _print_sample(s: Sample, tag: str):
        sqrts = _sqrt_s_ep(s.beam.ebeam1, s.beam.ebeam2)
        nan_lep = float(np.mean(~np.isfinite(s.lep_pt))) if s.lep_pt.size else 0.0
        nan_jet = float(np.mean(~np.isfinite(s.jet_pt))) if s.jet_pt.size else 0.0
        print(f"[{tag}] ebeam1={s.beam.ebeam1:.1f} GeV, ebeam2={s.beam.ebeam2:.1f} GeV, sqrt(s)~{sqrts:.3f} TeV, "
              f"sigma={s.sigma_pb:.6e} pb, N={s.n_events}")
        if nan_lep > 0 or nan_jet > 0:
            print(f"    WARNING: NaN fraction lep={nan_lep:.3e}, jet={nan_jet:.3e}  (selection may need tweaking)")
        if tag.lower().startswith("lhmuc") and s.mu_decay_is_highest_pt is not None:
            print(f"    LHmuC mu+ disambiguation: P(decay mu+ is highest-pT mu+) = {s.mu_decay_is_highest_pt:.3f}")

    _print_sample(lhec_sig, "LHeC signal")
    _print_sample(lhec_bkg, "LHeC bkg")
    _print_sample(lhmuc_sig, "LHmuC signal")
    _print_sample(lhmuc_bkg, "LHmuC bkg")

    sqrts_L = _sqrt_s_ep(lhec_sig.beam.ebeam1, lhec_sig.beam.ebeam2)
    sqrts_M = _sqrt_s_ep(lhmuc_sig.beam.ebeam1, lhmuc_sig.beam.ebeam2)

    # Bins
    max_lep = np.nanmax([np.nanmax(lhec_sig.lep_pt), np.nanmax(lhec_bkg.lep_pt),
                         np.nanmax(lhmuc_sig.lep_pt), np.nanmax(lhmuc_bkg.lep_pt)])
    max_jet = np.nanmax([np.nanmax(lhec_sig.jet_pt), np.nanmax(lhec_bkg.jet_pt),
                         np.nanmax(lhmuc_sig.jet_pt), np.nanmax(lhmuc_bkg.jet_pt)])
    lep_bins = build_default_bins(max_lep * 1.02, args.bin_step_lep)
    jet_bins = build_default_bins(max_jet * 1.02, args.bin_step_jet)

    # dσ/dpT plots
    cL, ds_L_sig = dsigma_hist(lhec_sig.lep_pt, lhec_sig.sigma_pb, lhec_sig.n_events, lep_bins)
    _,  ds_L_bkg = dsigma_hist(lhec_bkg.lep_pt, lhec_bkg.sigma_pb, lhec_bkg.n_events, lep_bins)
    cM, ds_M_sig = dsigma_hist(lhmuc_sig.lep_pt, lhmuc_sig.sigma_pb, lhmuc_sig.n_events, lep_bins)
    _,  ds_M_bkg = dsigma_hist(lhmuc_bkg.lep_pt, lhmuc_bkg.sigma_pb, lhmuc_bkg.n_events, lep_bins)

    cj, ds_LJ_sig = dsigma_hist(lhec_sig.jet_pt, lhec_sig.sigma_pb, lhec_sig.n_events, jet_bins)
    _,  ds_LJ_bkg = dsigma_hist(lhec_bkg.jet_pt, lhec_bkg.sigma_pb, lhec_bkg.n_events, jet_bins)
    _,  ds_MJ_sig = dsigma_hist(lhmuc_sig.jet_pt, lhmuc_sig.sigma_pb, lhmuc_sig.n_events, jet_bins)
    _,  ds_MJ_bkg = dsigma_hist(lhmuc_bkg.jet_pt, lhmuc_bkg.sigma_pb, lhmuc_bkg.n_events, jet_bins)

    plot_dsigma(
        args.outdir,
        title=rf"LHeC@{sqrts_L:.1f} TeV  Signal vs SM background  Observable: decay lepton",
        xlabel=r"$p_T^{\ell^+}$ [GeV]",
        centers=cL,
        ds_sig=ds_L_sig,
        ds_bkg=ds_L_bkg,
        lab_sig=rf"Signal ($\kappa={args.kappa0:g}$), $\sigma$={lhec_sig.sigma_pb:.3e} pb",
        lab_bkg=rf"SM background, $\sigma$={lhec_bkg.sigma_pb:.3e} pb",
        filename="LHeC_leppt_sig_vs_bkg.png",
    )
    plot_dsigma(
        args.outdir,
        title=rf"LHeC@{sqrts_L:.1f} TeV  Signal vs SM background  Observable: b (or leading jet)",
        xlabel=r"$p_T^{b\ \mathrm{or\ lead\ jet}}$ [GeV]",
        centers=cj,
        ds_sig=ds_LJ_sig,
        ds_bkg=ds_LJ_bkg,
        lab_sig=rf"Signal, $\sigma$={lhec_sig.sigma_pb:.3e} pb",
        lab_bkg=rf"SM background, $\sigma$={lhec_bkg.sigma_pb:.3e} pb",
        filename="LHeC_bpt_sig_vs_bkg.png",
    )
    plot_dsigma(
        args.outdir,
        title=rf"LH$\mu$C@{sqrts_M:.1f} TeV  Signal vs SM background  Observable: decay lepton",
        xlabel=r"$p_T^{\ell^+}$ [GeV]",
        centers=cM,
        ds_sig=ds_M_sig,
        ds_bkg=ds_M_bkg,
        lab_sig=rf"Signal ($\kappa={args.kappa0:g}$), $\sigma$={lhmuc_sig.sigma_pb:.3e} pb",
        lab_bkg=rf"SM background, $\sigma$={lhmuc_bkg.sigma_pb:.3e} pb",
        filename="LHmuC_leppt_sig_vs_bkg.png",
    )
    plot_dsigma(
        args.outdir,
        title=rf"LH$\mu$C@{sqrts_M:.1f} TeV  Signal vs SM background  Observable: b (or leading jet)",
        xlabel=r"$p_T^{b\ \mathrm{or\ lead\ jet}}$ [GeV]",
        centers=cj,
        ds_sig=ds_MJ_sig,
        ds_bkg=ds_MJ_bkg,
        lab_sig=rf"Signal, $\sigma$={lhmuc_sig.sigma_pb:.3e} pb",
        lab_bkg=rf"SM background, $\sigma$={lhmuc_bkg.sigma_pb:.3e} pb",
        filename="LHmuC_bpt_sig_vs_bkg.png",
    )

    # normalized signal-shape comparison
    cenS, densL = normalized_shape(lhec_sig.lep_pt, lep_bins)
    _,    densM = normalized_shape(lhmuc_sig.lep_pt, lep_bins)
    plot_shape(
        args.outdir,
        title=r"Signal shape (normalized): decay lepton $p_T$ tail  LHeC vs LH$\mu$C",
        xlabel=r"$p_T^{\ell^+}$ [GeV]",
        centers=cenS,
        dens_a=densL,
        dens_b=densM,
        lab_a=rf"LHeC@{sqrts_L:.1f} TeV (signal)",
        lab_b=rf"LH$\mu$C@{sqrts_M:.1f} TeV (signal)",
        filename="signal_shape_LHeC_vs_LHmuC_leppt.png",
    )

    # tail σ(pT>X)
    sig_tail_L = tail_sigma(lhec_sig.lep_pt, lhec_sig.sigma_pb, cuts)
    bkg_tail_L = tail_sigma(lhec_bkg.lep_pt, lhec_bkg.sigma_pb, cuts)
    sig_tail_M = tail_sigma(lhmuc_sig.lep_pt, lhmuc_sig.sigma_pb, cuts)
    bkg_tail_M = tail_sigma(lhmuc_bkg.lep_pt, lhmuc_bkg.sigma_pb, cuts)

    plot_tail_xsec(args.outdir, cuts, sig_tail_L, bkg_tail_L, sig_tail_M, bkg_tail_M,
                   filename="tail_xsec_leppt_LHeC_vs_LHmuC.png")

    # S/B and Z (at κ0)
    sb_L, Z_L = expected_sb_and_significance(sig_tail_L, bkg_tail_L, args.lumi, args.kappa0, deltaB=args.deltaB)
    sb_M, Z_M = expected_sb_and_significance(sig_tail_M, bkg_tail_M, args.lumi, args.kappa0, deltaB=args.deltaB)
    plot_sb_and_Z(args.outdir, cuts, sb_L, sb_M, Z_L, Z_M, args.lumi, args.deltaB)

    # κ95 vs cut
    k95_L = kappa95_cutcount(sig_tail_L, bkg_tail_L, args.lumi, args.kappa0, deltaB=args.deltaB)
    k95_M = kappa95_cutcount(sig_tail_M, bkg_tail_M, args.lumi, args.kappa0, deltaB=args.deltaB)
    plot_kappa_vs_cut(args.outdir, cuts, k95_L, k95_M, args.lumi, args.deltaB,
                      filename="kappa95_vs_tailcut_LHeC_vs_LHmuC.png")

    ratio = k95_M / k95_L
    plot_improvement_ratio(args.outdir, cuts, ratio,
                           filename="improvement_ratio_kappa95_LHmuC_over_LHeC.png")

    # 1D optimum summary
    def _best_1d(cuts_arr: np.ndarray, karr: np.ndarray) -> Tuple[float, float]:
        m = np.isfinite(karr)
        if not np.any(m):
            return float("nan"), float("nan")
        i = int(np.nanargmin(karr))
        return float(cuts_arr[i]), float(karr[i])

    best_cut_L, best_k_L = _best_1d(cuts, k95_L)
    best_cut_M, best_k_M = _best_1d(cuts, k95_M)
    print(f"\n[1D optimum @ L={args.lumi:.0f} fb^-1 using lepton tail]")
    print(f"   LHeC : best cut pT> {best_cut_L:.0f} GeV  -> k95={best_k_L:.3e}")
    print(f"   LHmuC: best cut pT> {best_cut_M:.0f} GeV  -> k95={best_k_M:.3e}")
    if np.isfinite(best_k_L) and np.isfinite(best_k_M):
        print(f"   ratio (LHmuC/LHeC) = {best_k_M/best_k_L:.3f}")

    # --------------------------
    # 2D optimization: (pTlep cut, pTjet cut)
    # --------------------------
    sig2d_L = tail_matrix_2d(lhec_sig.lep_pt, lhec_sig.jet_pt, lhec_sig.sigma_pb, cuts, jet_cuts)
    bkg2d_L = tail_matrix_2d(lhec_bkg.lep_pt, lhec_bkg.jet_pt, lhec_bkg.sigma_pb, cuts, jet_cuts)
    sig2d_M = tail_matrix_2d(lhmuc_sig.lep_pt, lhmuc_sig.jet_pt, lhmuc_sig.sigma_pb, cuts, jet_cuts)
    bkg2d_M = tail_matrix_2d(lhmuc_bkg.lep_pt, lhmuc_bkg.jet_pt, lhmuc_bkg.sigma_pb, cuts, jet_cuts)

    k2d_L = np.zeros_like(sig2d_L)
    k2d_M = np.zeros_like(sig2d_M)
    for ix in range(len(cuts)):
        k2d_L[ix, :] = kappa95_cutcount(sig2d_L[ix, :], bkg2d_L[ix, :], args.lumi, args.kappa0, deltaB=args.deltaB)
        k2d_M[ix, :] = kappa95_cutcount(sig2d_M[ix, :], bkg2d_M[ix, :], args.lumi, args.kappa0, deltaB=args.deltaB)

    plot_heatmap(
        args.outdir,
        title=rf"LHeC 2D cut optimization ($L$={args.lumi:.0f} fb$^{{-1}}$)  |  $\kappa_{{95}}$",
        xcuts=cuts, ycuts=jet_cuts, Z=k2d_L,
        xlabel=r"$p_T^{\ell^+} > X$ [GeV]",
        ylabel=r"$p_T^{j} > Y$ [GeV]",
        filename="heatmap_kappa95_LHeC_lepJet.png",
    )
    plot_heatmap(
        args.outdir,
        title=rf"LH$\mu$C 2D cut optimization ($L$={args.lumi:.0f} fb$^{{-1}}$)  |  $\kappa_{{95}}$",
        xcuts=cuts, ycuts=jet_cuts, Z=k2d_M,
        xlabel=r"$p_T^{\ell^+} > X$ [GeV]",
        ylabel=r"$p_T^{j} > Y$ [GeV]",
        filename="heatmap_kappa95_LHmuC_lepJet.png",
    )

    # best κ95 vs luminosity (2D grid)
    bestL = []
    bestM = []
    for Lfb in lumis:
        kk_L = np.zeros_like(sig2d_L)
        kk_M = np.zeros_like(sig2d_M)
        for ix in range(len(cuts)):
            kk_L[ix, :] = kappa95_cutcount(sig2d_L[ix, :], bkg2d_L[ix, :], Lfb, args.kappa0, deltaB=args.deltaB)
            kk_M[ix, :] = kappa95_cutcount(sig2d_M[ix, :], bkg2d_M[ix, :], Lfb, args.kappa0, deltaB=args.deltaB)
        bestL.append(float(np.nanmin(kk_L)))
        bestM.append(float(np.nanmin(kk_M)))

    bestL = np.array(bestL, dtype=float)
    bestM = np.array(bestM, dtype=float)

    plt.figure()
    plt.plot(lumis, bestL, marker="o", label="LHeC best (2D)")
    plt.plot(lumis, bestM, marker="o", label=r"LH$\mu$C best (2D)")
    plt.xscale("log"); plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.xlabel(r"Luminosity $L$ [fb$^{-1}$]")
    plt.ylabel(r"Best expected $\kappa_{95}$ (2D cuts)")
    plt.title("Optimized coupling reach vs luminosity (2D cuts)")
    plt.legend()
    _finish_plot(args.outdir, "best_kappa95_vs_lumi_2D.png")

    plt.figure()
    plt.plot(lumis, bestM / bestL, marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.xscale("log"); plt.yscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.xlabel(r"Luminosity $L$ [fb$^{-1}$]")
    plt.ylabel(r"Best $\kappa_{95}$(LH$\mu$C) / Best $\kappa_{95}$(LHeC)")
    plt.title("Optimized improvement factor vs luminosity (2D)\n(<1 means LHμC wins)")
    _finish_plot(args.outdir, "improvement_best_kappa95_vs_lumi_2D.png")

    # LHμC diagnostic: decay vs scattered mu+ pT
    if lhmuc_sig.mu_decay_pt.size > 0 and lhmuc_sig.mu_scatter_pt.size > 0:
        max_mu = float(np.nanmax([np.nanmax(lhmuc_sig.mu_decay_pt), np.nanmax(lhmuc_sig.mu_scatter_pt)]))
        mu_bins = build_default_bins(max_mu * 1.02, max(5.0, args.bin_step_lep))
        cen, d_dec = normalized_shape(lhmuc_sig.mu_decay_pt, mu_bins)
        _,   d_sca = normalized_shape(lhmuc_sig.mu_scatter_pt, mu_bins)

        plt.figure()
        plt.step(cen, d_dec, where="mid", label=r"decay $\mu^+$ (from $W^+$)")
        plt.step(cen, d_sca, where="mid", label=r"scattered $\mu^+$ (beam)")
        plt.yscale("log")
        plt.grid(True, which="both", linestyle=":")
        plt.xlabel(r"$p_T^{\mu^+}$ [GeV]")
        plt.ylabel(r"Normalized density [1/GeV]")
        plt.title(r"LH$\mu$C signal: $\mu^+$ disambiguation diagnostics")
        plt.legend()
        _finish_plot(args.outdir, "LHmuC_muplus_decay_vs_scatter_pt.png")

    print("\nSaved plots into:", args.outdir)

if __name__ == "__main__":
    main()

