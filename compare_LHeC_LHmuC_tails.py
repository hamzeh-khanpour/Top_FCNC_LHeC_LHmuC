import os, gzip, math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# USER INPUTS (edit here)
# -----------------------
FILES = {
    "LHeC": {
        "signal": "/home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHeC/Events/run_01/singletop_FCNC_LHeC.lhe",
        "bkg":    "/home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHeC_SM/Events/run_01/singletop_FCNC_LHeC_SM.lhe",
        "label":  r"LHeC@1.2 TeV",
    },
    "LHmuC": {
        "signal": "/home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHmuC/Events/run_01/singletop_FCNC_LHmuC.lhe",
        "bkg":    "/home/hamzeh-khanpour/MG5_aMC_v3_6_6/singletop_FCNC_LHmuC_SM/Events/run_01/singletop_FCNC_LHmuC_SM.lhe",
        "label":  r"LH$\mu$C@3.7 TeV",
    },
}

KAPPA0 = 1e-3  # your generated tqZ benchmark
LUMINOSITY_FB = 1000.0  # choose something (e.g. 1000 fb^-1 = 1 ab^-1)
DELTA_B = 0.0  # background fractional systematic (0.0 for stats-only; try 0.1 later)

# Binning (works for both; LHeC will just die off earlier)
BINS_LEP = np.array(list(np.linspace(0, 200, 41)) + list(np.linspace(200, 1000, 33)))
BINS_B   = np.array(list(np.linspace(0, 200, 41)) + list(np.linspace(200, 1000, 33)))

# Tail scan thresholds (GeV)
THR_SCAN = np.array([50,100,150,200,250,300,400,500,600,800,1000], dtype=float)

# -----------------------
# LHE utilities
# -----------------------
def _open(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")

def read_init_info(path):
    """
    Returns: (ebeam1, ebeam2, sigma_pb)
    sigma_pb from the 2nd line inside <init> block (first float).
    ebeam from the 1st line inside <init> block (3rd and 4th entries).
    """
    with _open(path) as f:
        in_init = False
        got_beams = False
        for line in f:
            if "<init>" in line:
                in_init = True
                continue
            if in_init and not got_beams:
                if line.strip() == "":
                    continue
                parts = line.split()
                # id1 id2 E1 E2 ...
                ebeam1 = float(parts[2]); ebeam2 = float(parts[3])
                got_beams = True
                continue
            if in_init and got_beams:
                if line.strip() == "":
                    continue
                parts = line.split()
                sigma_pb = float(parts[0])
                return ebeam1, ebeam2, sigma_pb
    raise RuntimeError(f"Could not find <init> info in {path}")

def pt(px, py):
    return math.hypot(px, py)

def parse_events_pts(path):
    """
    Extract:
      - pT of decay lepton l+ from W/top (e+ or mu+), robust even for mu+ beam
      - pT of b-quark from top decay if present; otherwise pT of leading jet (proxy)
    Returns:
      dict with keys: n_events, pt_lep_list, pt_b_list
    """
    pt_lep = []
    pt_b = []
    n_events = 0

    with _open(path) as f:
        in_event = False
        need_header = False
        nup = None
        particles = []

        for line in f:
            if "<event>" in line:
                in_event = True
                need_header = True
                particles = []
                continue

            if in_event and need_header:
                if line.strip() == "":
                    continue
                # NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP
                header = line.split()
                nup = int(header[0])
                need_header = False
                continue

            if in_event and (nup is not None) and len(particles) < nup:
                if line.strip() == "":
                    continue
                parts = line.split()
                # IDUP ISTUP MOTH1 MOTH2 COL1 COL2 Px Py Pz E M ...
                pid   = int(parts[0])
                ist   = int(parts[1])
                moth1 = int(parts[2])
                moth2 = int(parts[3])
                px    = float(parts[6])
                py    = float(parts[7])
                particles.append((pid, ist, moth1, moth2, px, py))
                continue

            if "</event>" in line and in_event:
                n_events += 1

                # Build arrays for quick lookup (LHE particle indices start at 1)
                pid_arr   = [None] + [p[0] for p in particles]
                ist_arr   = [None] + [p[1] for p in particles]
                moth1_arr = [None] + [p[2] for p in particles]
                px_arr    = [None] + [p[4] for p in particles]
                py_arr    = [None] + [p[5] for p in particles]

                # --- find decay lepton l+ (e+ or mu+) from W+ (pdg=24) ---
                # final-state positive leptons: e+ (pdg=-11) or mu+ (pdg=-13)
                cand_leps = []
                for i in range(1, len(pid_arr)):
                    if ist_arr[i] != 1:
                        continue
                    if pid_arr[i] in (-11, -13):  # e+ or mu+
                        cand_leps.append(i)

                def has_mother_Wplus(i):
                    m = moth1_arr[i]
                    if m is None or m <= 0:
                        return False
                    return pid_arr[m] == 24  # mother is W+

                decay_leps = [i for i in cand_leps if has_mother_Wplus(i)]
                if len(decay_leps) == 0 and len(cand_leps) > 0:
                    # fallback: if mother info weird, take the highest-pT positive lepton
                    i = max(cand_leps, key=lambda k: pt(px_arr[k], py_arr[k]))
                    pt_lep.append(pt(px_arr[i], py_arr[i]))
                elif len(decay_leps) > 0:
                    # if multiple (rare), take highest pT among W-derived
                    i = max(decay_leps, key=lambda k: pt(px_arr[k], py_arr[k]))
                    pt_lep.append(pt(px_arr[i], py_arr[i]))

                # --- find b from top decay if present (pdg=5) ---
                cand_bs = []
                for i in range(1, len(pid_arr)):
                    if ist_arr[i] != 1:
                        continue
                    if pid_arr[i] == 5:  # b
                        cand_bs.append(i)

                if len(cand_bs) > 0:
                    i = max(cand_bs, key=lambda k: pt(px_arr[k], py_arr[k]))
                    pt_b.append(pt(px_arr[i], py_arr[i]))
                else:
                    # fallback: take leading jet pT (quarks/gluons) as proxy
                    jets = []
                    for i in range(1, len(pid_arr)):
                        if ist_arr[i] != 1:
                            continue
                        if abs(pid_arr[i]) in (1,2,3,4,5) or pid_arr[i] == 21:
                            jets.append(i)
                    if len(jets) > 0:
                        i = max(jets, key=lambda k: pt(px_arr[k], py_arr[k]))
                        pt_b.append(pt(px_arr[i], py_arr[i]))

                # reset
                in_event = False
                nup = None
                continue

    return {"n_events": n_events, "pt_lep": np.array(pt_lep), "pt_b": np.array(pt_b)}

def dsigma_hist(pt_values, n_events_total, sigma_pb, bins):
    """
    Returns bin centers, dsigma/dpt in pb/GeV
    Uses weight per event = sigma / N_total, so integral matches sigma × acceptance.
    """
    widths = np.diff(bins)
    centers = 0.5*(bins[:-1] + bins[1:])
    if n_events_total <= 0:
        raise ValueError("n_events_total <= 0")
    w = sigma_pb / float(n_events_total)
    hist, _ = np.histogram(pt_values, bins=bins, weights=np.full(len(pt_values), w))
    return centers, hist / widths

def tail_xsec(pt_values, n_events_total, sigma_pb, thr):
    """
    Cross section (pb) for pT > thr.
    """
    if n_events_total <= 0:
        return 0.0
    frac = float(np.sum(pt_values > thr)) / float(n_events_total)
    return sigma_pb * frac

def s95_events(B_events):
    """
    Very simple 95% CL upper fluctuation estimate (Gaussian approx).
    For quick collider-to-collider comparison this is OK.
    """
    # if B very small, keep a floor to avoid nonsense
    if B_events <= 0:
        return 3.0  # ~few-event Poisson-ish conservative floor
    return 1.64 * math.sqrt(B_events)

# -----------------------
# Main
# -----------------------
def main():
    results = {}

    for collider, info in FILES.items():
        results[collider] = {}
        for sample in ("signal", "bkg"):
            path = info[sample]
            if not os.path.exists(path):
                raise FileNotFoundError(path)

            ebeam1, ebeam2, sigma_pb = read_init_info(path)
            pts = parse_events_pts(path)

            # print sanity
            sqs = math.sqrt(4.0 * ebeam1 * ebeam2) / 1000.0
            print(f"[{collider}][{sample}] ebeam1={ebeam1:.1f} GeV, ebeam2={ebeam2:.1f} GeV, sqrt(s)~{sqs:.3f} TeV, sigma={sigma_pb:.6e} pb, N={pts['n_events']}")

            results[collider][sample] = {
                "sigma_pb": sigma_pb,
                "ebeam1": ebeam1, "ebeam2": ebeam2, "sqs_TeV": sqs,
                "n_events": pts["n_events"],
                "pt_lep": pts["pt_lep"],
                "pt_b": pts["pt_b"],
            }

    # -----------------------
    # Plots: per collider, signal vs bkg
    # -----------------------
    for collider, info in FILES.items():
        lab = info["label"]

        # lepton
        plt.figure()
        for sample, ls in [("signal","-"), ("bkg","-")]:
            d = results[collider][sample]
            x, y = dsigma_hist(d["pt_lep"], d["n_events"], d["sigma_pb"], BINS_LEP)
            plt.step(x, y, where="mid", linestyle=ls,
                     label=f"{sample.upper()}, σ={d['sigma_pb']:.3e} pb")
        plt.yscale("log")
        plt.xlabel(r"$p_T^{\ell^+}$ [GeV]")
        plt.ylabel(r"$d\sigma/dp_T$ [pb/GeV]")
        plt.title(f"{lab}\nSignal (tqZ={KAPPA0:.0e}) vs SM background\nObservable: decay lepton")
        plt.legend()
        plt.grid(True, which="both", linestyle=":")
        plt.tight_layout()
        plt.savefig(f"{collider}_leppt_sig_vs_bkg.png", dpi=200)

        # b (or jet proxy)
        plt.figure()
        for sample, ls in [("signal","-"), ("bkg","-")]:
            d = results[collider][sample]
            x, y = dsigma_hist(d["pt_b"], d["n_events"], d["sigma_pb"], BINS_B)
            plt.step(x, y, where="mid", linestyle=ls,
                     label=f"{sample.upper()}, σ={d['sigma_pb']:.3e} pb")
        plt.yscale("log")
        plt.xlabel(r"$p_T^{b\ \mathrm{(or\ leading\ jet)}}$ [GeV]")
        plt.ylabel(r"$d\sigma/dp_T$ [pb/GeV]")
        plt.title(f"{lab}\nSignal (tqZ={KAPPA0:.0e}) vs SM background\nObservable: b (or leading jet)")
        plt.legend()
        plt.grid(True, which="both", linestyle=":")
        plt.tight_layout()
        plt.savefig(f"{collider}_bpt_sig_vs_bkg.png", dpi=200)

    # -----------------------
    # Plot: signal shape comparison LHeC vs LHmuC (normalized)
    # -----------------------
    plt.figure()
    for collider, info in FILES.items():
        d = results[collider]["signal"]
        # normalized shape: use dsigma but divide by total sigma
        x, y = dsigma_hist(d["pt_lep"], d["n_events"], d["sigma_pb"], BINS_LEP)
        y_norm = y / (np.sum(y * np.diff(BINS_LEP)) + 1e-30)
        plt.step(x, y_norm, where="mid", label=f"{info['label']} (signal)")
    plt.yscale("log")
    plt.xlabel(r"$p_T^{\ell^+}$ [GeV]")
    plt.ylabel(r"Normalized $(1/\sigma)\,d\sigma/dp_T$ [1/GeV]")
    plt.title("Signal shape comparison (normalized)\nDecay lepton pT tail: LHeC vs LHμC")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig("signal_shape_LHeC_vs_LHmuC_leppt.png", dpi=200)

    # -----------------------
    # Tail scan: S/B and expected kappa95 vs threshold
    # -----------------------
    L_pb_inv = LUMINOSITY_FB * 1e3  # fb^-1 -> pb^-1

    plt.figure()
    for collider, info in FILES.items():
        sig = results[collider]["signal"]
        bkg = results[collider]["bkg"]

        SB = []
        k95 = []

        for thr in THR_SCAN:
            sig_tail = tail_xsec(sig["pt_lep"], sig["n_events"], sig["sigma_pb"], thr)
            bkg_tail = tail_xsec(bkg["pt_lep"], bkg["n_events"], bkg["sigma_pb"], thr)

            # event yields
            S = sig_tail * L_pb_inv
            B = bkg_tail * L_pb_inv

            # include simple background systematic
            denom = math.sqrt(B + (DELTA_B * B)**2) if B > 0 else 1.0
            SB.append((S / B) if B > 0 else 0.0)

            # crude S95 and kappa limit
            S95 = s95_events(B)  # events
            if S > 0:
                kappa95 = KAPPA0 * math.sqrt(S95 / S)
            else:
                kappa95 = np.nan
            k95.append(kappa95)

        plt.plot(THR_SCAN, k95, marker="o", label=f"{info['label']} (κ95 from tail)")

    plt.yscale("log")
    plt.xlabel(r"Tail cut: $p_T^{\ell^+} > X$ [GeV]")
    plt.ylabel(r"Expected limit on coupling $\kappa_{95}$ (arb. Gaussian approx)")
    plt.title(f"Expected coupling reach vs tail cut (using $p_T^{{\\ell^+}}$)\nL={LUMINOSITY_FB:.0f} fb$^{{-1}}$, δB={DELTA_B:.2f}")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig("kappa95_vs_tailcut_LHeC_vs_LHmuC.png", dpi=200)

    print("\nSaved PNGs:")
    print("  * LHeC_leppt_sig_vs_bkg.png, LHeC_bpt_sig_vs_bkg.png")
    print("  * LHmuC_leppt_sig_vs_bkg.png, LHmuC_bpt_sig_vs_bkg.png")
    print("  * signal_shape_LHeC_vs_LHmuC_leppt.png")
    print("  * kappa95_vs_tailcut_LHeC_vs_LHmuC.png")

if __name__ == "__main__":
    main()

