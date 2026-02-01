import os, gzip, math
import numpy as np
import matplotlib.pyplot as plt


import mplhep as hep
# Publication style
hep.style.use("CMS")


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

KAPPA0 = 1e-3
LUMINOSITY_FB = 1000.0      # 1 ab^-1
DELTA_B = 0.0               # try 0.05 or 0.10 later

def make_bins():
    # no duplicated edges!
    return np.concatenate([np.linspace(0, 200, 41), np.linspace(200, 1000, 33)[1:]])

BINS_LEP = make_bins()
BINS_B   = make_bins()

THR_SCAN = np.array([50,100,150,200,250,300,350,400,500,600,800,1000], dtype=float)

def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")

def read_init_info(path):
    with _open(path) as f:
        in_init = False
        got_beams = False
        for line in f:
            if "<init>" in line:
                in_init = True
                continue
            if in_init and not got_beams:
                if not line.strip():
                    continue
                parts = line.split()
                ebeam1 = float(parts[2]); ebeam2 = float(parts[3])
                got_beams = True
                continue
            if in_init and got_beams:
                if not line.strip():
                    continue
                parts = line.split()
                sigma_pb = float(parts[0])
                return ebeam1, ebeam2, sigma_pb
    raise RuntimeError(f"Could not find <init> info in {path}")

def pt(px, py): return math.hypot(px, py)

def parse_events_pts(path):
    pt_lep, pt_b = [], []
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
                if not line.strip():
                    continue
                header = line.split()
                nup = int(header[0])
                need_header = False
                continue

            if in_event and (nup is not None) and len(particles) < nup:
                if not line.strip():
                    continue
                parts = line.split()
                pid   = int(parts[0])
                ist   = int(parts[1])
                moth1 = int(parts[2])
                pxv   = float(parts[6])
                pyv   = float(parts[7])
                particles.append((pid, ist, moth1, pxv, pyv))
                continue

            if "</event>" in line and in_event:
                n_events += 1

                pid_arr   = [None] + [p[0] for p in particles]
                ist_arr   = [None] + [p[1] for p in particles]
                moth1_arr = [None] + [p[2] for p in particles]
                px_arr    = [None] + [p[3] for p in particles]
                py_arr    = [None] + [p[4] for p in particles]

                # decay lepton: e+ or mu+ (pdg -11, -13), ideally from W+ (pdg 24)
                cand_leps = [i for i in range(1, len(pid_arr))
                             if ist_arr[i]==1 and pid_arr[i] in (-11, -13)]
                def from_Wplus(i):
                    m = moth1_arr[i]
                    return (m is not None and m>0 and pid_arr[m]==24)

                leps = [i for i in cand_leps if from_Wplus(i)]
                if len(leps)==0 and len(cand_leps)>0:
                    leps = cand_leps  # fallback
                if len(leps)>0:
                    i = max(leps, key=lambda k: pt(px_arr[k], py_arr[k]))
                    pt_lep.append(pt(px_arr[i], py_arr[i]))

                # b from top decay if present: final b with mother top (pdg 6)
                cand_bs = [i for i in range(1, len(pid_arr))
                           if ist_arr[i]==1 and pid_arr[i]==5]
                def from_top(i):
                    m = moth1_arr[i]
                    return (m is not None and m>0 and pid_arr[m]==6)
                bs = [i for i in cand_bs if from_top(i)]
                if len(bs)==0:
                    bs = cand_bs
                if len(bs)>0:
                    i = max(bs, key=lambda k: pt(px_arr[k], py_arr[k]))
                    pt_b.append(pt(px_arr[i], py_arr[i]))

                in_event = False
                nup = None

    return {"n_events": n_events, "pt_lep": np.array(pt_lep), "pt_b": np.array(pt_b)}

def dsigma_hist(pt_values, n_events_total, sigma_pb, bins):
    widths  = np.diff(bins)
    centers = 0.5*(bins[:-1] + bins[1:])
    w = sigma_pb / float(n_events_total)
    hist, _ = np.histogram(pt_values, bins=bins, weights=np.full(len(pt_values), w))
    y = np.zeros_like(hist, dtype=float)
    m = widths > 0
    y[m] = hist[m] / widths[m]
    y[~m] = np.nan
    return centers, y

def tail_xsec(pt_values, n_events_total, sigma_pb, thr):
    # sigma/N_total * N_pass
    if n_events_total <= 0:
        return 0.0
    n_pass = float(np.sum(pt_values > thr))
    return sigma_pb * (n_pass / float(n_events_total))

def s95_events(B):
    # crude 95% CL "upper fluctuation"
    if B <= 0:
        return 3.0
    return 1.64 * math.sqrt(B)

def safe_step(x, y, **kw):
    y2 = np.array(y, dtype=float)
    y2[y2 <= 0] = np.nan
    plt.step(x, y2, where="mid", **kw)

def main():
    results = {}
    for collider, info in FILES.items():
        results[collider] = {}
        for sample in ("signal","bkg"):
            path = info[sample]
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            ebeam1, ebeam2, sigma_pb = read_init_info(path)
            pts = parse_events_pts(path)
            sqs = math.sqrt(4.0*ebeam1*ebeam2)/1000.0
            print(f"[{collider}][{sample}] sqrt(s)~{sqs:.3f} TeV, sigma={sigma_pb:.6e} pb, N={pts['n_events']}, Nlep={len(pts['pt_lep'])}, Nb={len(pts['pt_b'])}")
            results[collider][sample] = dict(
                sigma_pb=sigma_pb, n_events=pts["n_events"],
                pt_lep=pts["pt_lep"], pt_b=pts["pt_b"],
                sqs_TeV=sqs
            )

    # (1) Signal shape comparison: LHeC vs LHmuC (normalized)
    plt.figure()
    for collider, info in FILES.items():
        d = results[collider]["signal"]
        x, y = dsigma_hist(d["pt_lep"], d["n_events"], d["sigma_pb"], BINS_LEP)
        area = np.nansum(y * np.diff(BINS_LEP))
        y_norm = y / area if area>0 else y
        safe_step(x, y_norm, label=f"{info['label']} (signal)")
    plt.yscale("log")
    plt.xlabel(r"$p_T^{\ell^+}$ [GeV]")
    plt.ylabel(r"Normalized $(1/\sigma)\,d\sigma/dp_T$ [1/GeV]")
    plt.title("Signal shape (normalized): decay lepton tail\nLHeC vs LHμC")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig("signal_shape_LHeC_vs_LHmuC_leppt_v2.png", dpi=200)

    # (2) Cumulative tail cross section sigma(pT > X): signal and background
    plt.figure()
    for collider, info in FILES.items():
        for sample, ls in [("signal","-"), ("bkg","--")]:
            d = results[collider][sample]
            sig_tail = [tail_xsec(d["pt_lep"], d["n_events"], d["sigma_pb"], thr) for thr in THR_SCAN]
            plt.plot(THR_SCAN, sig_tail, marker="o", linestyle=ls,
                     label=f"{info['label']} {sample}")
    plt.yscale("log")
    plt.xlabel(r"Tail cut: $p_T^{\ell^+} > X$ [GeV]")
    plt.ylabel(r"$\sigma(p_T^{\ell^+} > X)$ [pb]")
    plt.title("Cumulative tail cross sections (monotonic)\nSolid=signal, dashed=background")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig("tail_xsec_leppt_LHeC_vs_LHmuC.png", dpi=200)

    # (3) Expected kappa95 vs tail cut + improvement ratio
    L_pb_inv = LUMINOSITY_FB * 1e3

    k95 = {}
    plt.figure()
    for collider, info in FILES.items():
        sig = results[collider]["signal"]
        bkg = results[collider]["bkg"]
        k95[collider] = []
        for thr in THR_SCAN:
            sig_tail = tail_xsec(sig["pt_lep"], sig["n_events"], sig["sigma_pb"], thr)
            bkg_tail = tail_xsec(bkg["pt_lep"], bkg["n_events"], bkg["sigma_pb"], thr)
            S = sig_tail * L_pb_inv
            B = bkg_tail * L_pb_inv
            S95 = s95_events(B)
            denom = S if S>0 else np.nan
            kappa95 = KAPPA0 * math.sqrt(S95/denom) if denom==denom else np.nan
            k95[collider].append(kappa95)
        plt.plot(THR_SCAN, k95[collider], marker="o", label=f"{info['label']}")

    plt.yscale("log")
    plt.xlabel(r"Tail cut: $p_T^{\ell^+} > X$ [GeV]")
    plt.ylabel(r"Expected $\kappa_{95}$ (stats-only approx)")
    plt.title(f"Coupling reach vs tail cut (using $p_T^{{\\ell^+}}$)\nL={LUMINOSITY_FB:.0f} fb$^{{-1}}$, δB={DELTA_B:.2f}")
    plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig("kappa95_vs_tailcut_LHeC_vs_LHmuC_v2.png", dpi=200)

    # improvement ratio plot
    plt.figure()
    ratio = np.array(k95["LHmuC"]) / np.array(k95["LHeC"])
    plt.plot(THR_SCAN, ratio, marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.yscale("log")
    plt.xlabel(r"Tail cut: $p_T^{\ell^+} > X$ [GeV]")
    plt.ylabel(r"$\kappa_{95}(\mathrm{LH\mu C}) / \kappa_{95}(\mathrm{LHeC})$")
    plt.title("Improvement factor (<1 means LHμC gives stronger limit)")
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig("improvement_ratio_kappa95_LHmuC_over_LHeC.png", dpi=200)

    print("\nSaved:")
    print("  signal_shape_LHeC_vs_LHmuC_leppt_v2.png")
    print("  tail_xsec_leppt_LHeC_vs_LHmuC.png")
    print("  kappa95_vs_tailcut_LHeC_vs_LHmuC_v2.png")
    print("  improvement_ratio_kappa95_LHmuC_over_LHeC.png")

if __name__ == "__main__":
    main()

