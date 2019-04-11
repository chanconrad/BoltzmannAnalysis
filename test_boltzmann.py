import boltzmann_analysis
import matplotlib.pyplot as plt
import numpy as np

r = boltzmann_analysis.BoltzmannRun('/Users/conrad/repos/Boltzmann/output/')

# r.plot_spatial('number_luminosity', 20)
r.plot_spatial('number_luminosity_lab', 20)
# r.plot_spatial('number_flux_density_radial', 20)
# r.plot_spatial('cross_section_r', 20)
# r.plot_spatial('energy_luminosity_lab', 20)
# r.plot_spectrum(20)
# r.plot_moments(20, show_exact=True)
# r.plot_moments(20, show_exact=False)
# plt.xscale('log')
# r.plot_flux_factors(20, show_exact=True)

# r.plot_mu_distribution(20, flavour=0)
# plt.xscale('log')
# plt.ylim(0.0, 1.0)
# plt.xlim(0.0, 0.05)

# plt.figure()
# r.plot_mu_distribution(0, flavour=0)
# plt.ylim(-1,1)
# plt.figure()
# r.plot_mu_distribution(2, flavour=0)
# plt.ylim(0,1)
# plt.figure()
# r.plot_mu_distribution(4, flavour=0)
# plt.ylim(0,1)
# plt.figure()
# r.plot_mu_distribution(6, flavour=0)
# plt.ylim(0,1)
# plt.figure()
# r.plot_mu_distribution(8, flavour=0)
# plt.ylim(0,1)
# plt.figure()
# r.plot_mu_distribution(12, flavour=0)
# plt.ylim(0,1)
# plt.xscale('log')
# #
# plt.figure()
# r.plot_mu_distribution(20, flavour=0)
# plt.xscale('log')

# plt.xscale('log')

#
# plt.figure()
# r.plot_mu_distribution(20, flavour=1)
#
# plt.figure()
# r.plot_mu_distribution(20, flavour=2)


# radius = r[20].r
# volume = r[20].volume[:,0,0]
# area_r = r[20].area_r[1:,0,0]
# f = r[20].f[:,0,0,2,6,2]
#
# plt.plot(radius, volume / area_r / radius)

r.close()

# plt.ylim(-1.e-5,1.e-5)
plt.show()
