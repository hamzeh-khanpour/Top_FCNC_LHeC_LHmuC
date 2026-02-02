import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Coefficients (LHeC @ 1.2 TeV)
# -----------------------------------
a_tqZ = 2.91      # pb
a_tqgamma = 13.33 # pb

# -----------------------------------
# Coupling range
# -----------------------------------
kappa = np.linspace(0.0, 0.1, 500)

# -----------------------------------
# FCNC cross sections
# -----------------------------------
sigma_tqZ = a_tqZ * kappa**2
sigma_tqgamma = a_tqgamma * kappa**2

# -----------------------------------
# Plot: sigma vs kappa
# -----------------------------------
plt.figure(figsize=(7.5, 5.5))

plt.plot(
    kappa, sigma_tqZ,
    lw=2, label=r'$tqZ$'
)

plt.plot(
    kappa, sigma_tqgamma,
    lw=2, ls='--', label=r'$tq\gamma$'
)

plt.xlabel(r'Anomalous coupling $\kappa$', fontsize=13)
plt.ylabel(r'$\sigma_{\mathrm{FCNC}}$ [pb]', fontsize=13)

plt.title(
    r'LHeC@1.2 TeV: $\sigma_{\mathrm{FCNC}}(\kappa) = a\,\kappa^2$',
    fontsize=14
)

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(
    'LHeC_sigma_vs_kappa_tqZ_tqgamma.png',
    dpi=300
)
plt.show()
