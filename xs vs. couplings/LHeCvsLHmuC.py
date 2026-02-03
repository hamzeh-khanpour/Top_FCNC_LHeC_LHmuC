import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Coefficients a extracted from MG5 (pb)
# --------------------------------------------------
# LHeC @ 1.2 TeV
a_LHeC_tqZ = 2.91
a_LHeC_tqgamma = 13.33

# LHmuC @ 3.7 TeV
a_LHmuC_tqZ = 8.387
a_LHmuC_tqgamma = 40.74

# --------------------------------------------------
# Coupling range (linear scale: 10^-5 to 10^-1)
# --------------------------------------------------
kappa = np.linspace(1e-5, 1e-1, 500)

# --------------------------------------------------
# Cross sections: sigma = a * kappa^2
# --------------------------------------------------
sigma_LHeC_tqZ = a_LHeC_tqZ * kappa**2
sigma_LHeC_tqgamma = a_LHeC_tqgamma * kappa**2

sigma_LHmuC_tqZ = a_LHmuC_tqZ * kappa**2
sigma_LHmuC_tqgamma = a_LHmuC_tqgamma * kappa**2

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(8, 6))

plt.plot(kappa, sigma_LHeC_tqZ,
         lw=2, label=r'LHeC@1.2 TeV ($tqZ$)')

plt.plot(kappa, sigma_LHeC_tqgamma,
         lw=2, ls='--', label=r'LHeC@1.2 TeV ($tq\gamma$)')

plt.plot(kappa, sigma_LHmuC_tqZ,
         lw=2, label=r'LH$\mu$C@3.7 TeV ($tqZ$)')

plt.plot(kappa, sigma_LHmuC_tqgamma,
         lw=2, ls='--', label=r'LH$\mu$C@3.7 TeV ($tq\gamma$)')

plt.yscale('log')
plt.ylim(1e-6, 1e0)

plt.xlabel(r'Anomalous coupling $\kappa$', fontsize=13)
plt.ylabel(r'$\sigma_{\mathrm{FCNC}}$ [pb]', fontsize=13)

plt.title(
    r'FCNC single-top production: $\sigma_{\mathrm{FCNC}}(\kappa)=a\,\kappa^2$',
    fontsize=14
)

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig(
    'sigma_vs_kappa_LHeC_vs_LHmuC.png',
    dpi=300
)
plt.show()
