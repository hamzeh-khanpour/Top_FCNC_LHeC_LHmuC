#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import gzip
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Simple TLorentzVector-like 4-vector
# -----------------------------
class FourVec:
    __slots__ = ("px", "py", "pz", "E")

    def __init__(self, px: float, py: float, pz: float, E: float):
        self.px = px
        self.py = py
        self.pz = pz
        self.E  = E

    @property
    def pt(self) -> float:
        return math.hypot(self.px, self.py)


def open_text(path: str):
    """Open .lhe or .lhe.gz transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def extract_xsec_pb_from_header(path: str) -> Optional[float]:
    """
    Try to extract total cross section from:
      1) <MGGenerationInfo> 'Integrated weight (pb)  :   ...'
      2) <init> second line: first float is xsec (pb) in MG5 LHE
    Returns xsec in pb or None if not found.
    """
    xsec = None
    in_init = False
    init_payload_lines: List[str] = []

    # Regex for floats incl scientific notation
    float_re = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")

    with open_text(path) as f:
        for line in f:
            # 1) MGGenerationInfo integrated weight
            if "Integrated weight (pb)" in line:
                nums = float_re.findall(line)
                if nums:
                    try:
                        xsec = float(nums[0])
                        return xsec
                    except ValueError:
                        pass

            # 2) init block (grab first two non-empty lines inside <init>)
            if "<init>" in line:
                in_init = True
                init_payload_lines = []
                continue

            if in_init:
                if "</init>" in line:
                    in_init = False
                    # In MG5, the 2nd line after beam line is typically: xsec  xerr  xmax  lpr
                    if len(init_payload_lines) >= 2:
                        parts = init_payload_lines[1].split()
                        try:
                            return float(parts[0])
                        except Exception:
                            return xsec
                    continue

                stripped = line.strip()
                if stripped and (not stripped.startswith("#")):
                    init_payload_lines.append(stripped)

            # Stop early once we hit first event tag and we already have xsec
            if "<event>" in line:
                break

    return xsec


def parse_lhe_observables(
    path: str,
    sample_kind: str,
) -> Dict[str, object]:
    """
    Parse LHE and return arrays for:
      - pt_lplus: pT of positive charged lepton (e+ or mu+) in final state
      - pt_blike: pT of b-jet-like object:
           * for signal: b quark (pdg=5)
           * for background: leading parton jet (quark/gluon) (mistag proxy)
    Also returns:
      - xsec_pb: total cross section in pb (from header)
      - n_events_used_{...}
    """
    assert sample_kind in ("signal", "background")

    xsec_pb = extract_xsec_pb_from_header(path)
    if xsec_pb is None:
        raise RuntimeError(f"Could not find cross section in header for: {path}")

    pt_lplus: List[float] = []
    w_lplus:  List[float] = []

    pt_blike: List[float] = []
    w_blike:  List[float] = []

    n_events_total = 0
    n_events_with_lplus = 0
    n_events_with_blike = 0

    with open_text(path) as f:
        it = iter(f)
        for line in it:
            if "<event>" not in line:
                continue

            # Event header line
            header = next(it).strip()
            if not header:
                continue
            # format: NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP
            parts = header.split()
            if len(parts) < 3:
                continue

            nup = int(parts[0])
            # event weight (not necessarily normalized); we use it for fractions only
            try:
                wgt = float(parts[2])
            except Exception:
                wgt = 1.0

            n_events_total += 1

            # Collect final-state particles
            leptons_plus: List[Tuple[float, FourVec]] = []
            b_quarks: List[Tuple[float, FourVec]] = []
            jets: List[Tuple[float, FourVec]] = []

            for _ in range(nup):
                pline = next(it).strip()
                if not pline:
                    continue
                cols = pline.split()
                if len(cols) < 10:
                    continue

                pdg = int(cols[0])
                status = int(cols[1])

                # px py pz E are cols[6:10]
                px = float(cols[6]); py = float(cols[7]); pz = float(cols[8]); E = float(cols[9])
                v = FourVec(px, py, pz, E)

                # Only final state objects
                if status != 1:
                    continue

                # Positive charged leptons: e+ (-11), mu+ (-13)
                if pdg in (-11, -13):
                    leptons_plus.append((v.pt, v))

                # b-quark
                if pdg == 5:
                    b_quarks.append((v.pt, v))

                # jets: quarks (|pdg|<=5) or gluon (21)
                # exclude leptons/neutrinos/photons/etc
                if (pdg == 21) or (1 <= abs(pdg) <= 5):
                    jets.append((v.pt, v))

            # Skip to </event>
            # (There can be extra lines like weights; consume until we hit </event>)
            for line2 in it:
                if "</event>" in line2:
                    break

            # Choose l+ (highest pT if multiple)
            if leptons_plus:
                leptons_plus.sort(key=lambda x: x[0], reverse=True)
                pt_lplus.append(leptons_plus[0][0])
                w_lplus.append(wgt)
                n_events_with_lplus += 1

            # Choose b-like
            if sample_kind == "signal":
                if b_quarks:
                    b_quarks.sort(key=lambda x: x[0], reverse=True)
                    pt_blike.append(b_quarks[0][0])
                    w_blike.append(wgt)
                    n_events_with_blike += 1
            else:
                # background: use leading jet as "b-jet mistag proxy"
                if jets:
                    jets.sort(key=lambda x: x[0], reverse=True)
                    pt_blike.append(jets[0][0])
                    w_blike.append(wgt)
                    n_events_with_blike += 1

    return {
        "path": path,
        "xsec_pb": float(xsec_pb),
        "pt_lplus": np.asarray(pt_lplus, dtype=float),
        "w_lplus": np.asarray(w_lplus, dtype=float),
        "pt_blike": np.asarray(pt_blike, dtype=float),
        "w_blike": np.asarray(w_blike, dtype=float),
        "n_events_total": n_events_total,
        "n_events_with_lplus": n_events_with_lplus,
        "n_events_with_blike": n_events_with_blike,
    }


def dsigma_over_pt(
    pt: np.ndarray,
    w: np.ndarray,
    xsec_pb: float,
    bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dσ/dpT in pb/GeV using:
      dσ_bin = xsec * (sum_w_bin / sum_w_total)
      dσ/dpT = dσ_bin / bin_width
    """
    if len(pt) == 0:
        hist = np.zeros(len(bins) - 1)
        centers = 0.5 * (bins[1:] + bins[:-1])
        return centers, hist

    sum_w = float(np.sum(w)) if len(w) else float(len(pt))
    if sum_w <= 0:
        sum_w = float(len(pt))

    hist_w, edges = np.histogram(pt, bins=bins, weights=w)
    frac = hist_w / sum_w
    widths = np.diff(edges)
    dsig = xsec_pb * frac / widths
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, dsig


def make_bins_from_data(a: np.ndarray, b: np.ndarray, nbins: int, xmin: float = 0.0) -> np.ndarray:
    mx = 0.0
    if len(a): mx = max(mx, float(np.max(a)))
    if len(b): mx = max(mx, float(np.max(b)))
    mx = max(mx, xmin + 1.0)
    # Round up to a nice value
    xmax = math.ceil(mx / 10.0) * 10.0
    return np.linspace(xmin, xmax, nbins + 1)


def plot_overlay(
    x1, y1, lab1,
    x2, y2, lab2,
    xlabel, ylabel, title, outpath
):
    plt.figure()
    plt.step(x1, y1, where="mid", label=lab1)
    plt.step(x2, y2, where="mid", label=lab2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale("log")  # tails are easier to see on log-scale; comment out if you prefer linear
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print(f"[saved] {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Compare dsigma/dpT from two LHE files (signal vs background).")
    parser.add_argument(
        "--signal",
        default="/home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHeC/Events/run_01/singletop_FCNC_LHeC.lhe",
        help="Path to signal LHE (.lhe or .lhe.gz)"
    )
    parser.add_argument(
        "--background",
        default="/home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHeC_SM/Events/run_01/singletop_FCNC_LHeC_SM.lhe",
        help="Path to background LHE (.lhe or .lhe.gz)"
    )
    parser.add_argument("--nbins", type=int, default=50, help="Number of bins")
    parser.add_argument("--outdir", default="plots_LHeC", help="Output directory for _LHeC")
    parser.add_argument(
        "--b_mistag",
        type=float,
        default=1.0,
        help="Optional multiplicative factor for background 'b-jet' (mistag) rate, e.g. 0.01. Default=1."
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Read samples ---
    sig = parse_lhe_observables(args.signal, sample_kind="signal")
    bkg = parse_lhe_observables(args.background, sample_kind="background")

    print("\n=== Summary ===")
    print(f"Signal      : {sig['path']}")
    print(f"  xsec (pb)  : {sig['xsec_pb']:.6e}")
    print(f"  events     : {sig['n_events_total']}  (with l+={sig['n_events_with_lplus']}, with b={sig['n_events_with_blike']})")
    print(f"Background  : {bkg['path']}")
    print(f"  xsec (pb)  : {bkg['xsec_pb']:.6e}")
    print(f"  events     : {bkg['n_events_total']}  (with l+={bkg['n_events_with_lplus']}, with jet={bkg['n_events_with_blike']})")
    print("==============\n")

    # --- Binning ---
    bins_l = make_bins_from_data(sig["pt_lplus"], bkg["pt_lplus"], nbins=args.nbins, xmin=0.0)
    bins_b = make_bins_from_data(sig["pt_blike"], bkg["pt_blike"], nbins=args.nbins, xmin=0.0)

    # --- dσ/dpT ---
    x_l_sig, y_l_sig = dsigma_over_pt(sig["pt_lplus"], sig["w_lplus"], sig["xsec_pb"], bins_l)
    x_l_bkg, y_l_bkg = dsigma_over_pt(bkg["pt_lplus"], bkg["w_lplus"], bkg["xsec_pb"], bins_l)

    x_b_sig, y_b_sig = dsigma_over_pt(sig["pt_blike"], sig["w_blike"], sig["xsec_pb"], bins_b)
    x_b_bkg, y_b_bkg = dsigma_over_pt(bkg["pt_blike"], bkg["w_blike"], bkg["xsec_pb"] * args.b_mistag, bins_b)

    # --- Labels ---
    title_common = "LHeC@1.2 TeV\nSignal (tqZ = $10^{-3}$) vs SM background"

    lab_sig = f"Signal tqZ ($10^{{-3}}$),  σ={sig['xsec_pb']:.3e} pb"
    lab_bkg = f"SM background,  σ={bkg['xsec_pb']:.3e} pb"
    if args.b_mistag != 1.0:
        lab_bkg += f"  × mistag({args.b_mistag:g})"

    # --- Plot 1: lepton pT ---
    plot_overlay(
        x_l_sig, y_l_sig, lab_sig,
        x_l_bkg, y_l_bkg, lab_bkg,
        xlabel=r"$p_T^{\ell^+}\ \mathrm{[GeV]}$",
        ylabel=r"$d\sigma/dp_T\ \mathrm{[pb/GeV]}$",
        title=title_common + "\nObservable: lepton from decay",
        outpath=os.path.join(args.outdir, "dsigma_dpt_lepton.png"),
    )

    # --- Plot 2: b-jet pT (signal b-quark vs background leading jet as mistag proxy) ---
    plot_overlay(
        x_b_sig, y_b_sig, lab_sig + " (b from top)",
        x_b_bkg, y_b_bkg, lab_bkg + " (leading jet → b-mistag proxy)",
        xlabel=r"$p_T^{b\mathrm{-jet}}\ \mathrm{[GeV]}$",
        ylabel=r"$d\sigma/dp_T\ \mathrm{[pb/GeV]}$",
        title=title_common + "\nObservable: b-jet (signal) / jet-as-b (background)",
        outpath=os.path.join(args.outdir, "dsigma_dpt_bjet.png"),
    )


if __name__ == "__main__":
    main()

