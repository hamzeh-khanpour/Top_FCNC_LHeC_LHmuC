import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Input data (LHeC @ 1.2 TeV)
# -----------------------------------
kappa = np.array([1e-4, 1e-3, 1e-2, 1e-1])

sigma_tqZ = np.array([
    2.911e-08,
    2.9109e-06,
    2.908e-04,
    2.91e-02
])  # pb

sigma_tqgamma = np.array([
    1.332e-07,
    1.33375e-05,
    1.332e-03,
    1.333e-01
])  # pb

# -----------------------------------
# Plot
# -----------------------------------
plt.figure(figsize=(7.5, 5.5))

plt.plot(
    kappa, sigma_tqZ,
    marker='o', lw=2,
    label=r'$tqZ$'
)

plt.plot(
    kappa, sigma_tqgamma,
    marker='s', lw=2, ls='--',
    label=r'$tq\gamma$'
)

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'FCNC coupling', fontsize=13)
plt.ylabel(r'$\sigma$(LHeC@1.2 TeV) [pb]', fontsize=13)

plt.title(
    r'LHeC@1.2 TeV: FCNC single-top production',
    fontsize=14
)

plt.grid(True, which='both', linestyle=':', alpha=0.7)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(
    'LHeC_xsec_vs_coupling_tqZ_tqgamma.png',
    dpi=300
)
plt.show()
