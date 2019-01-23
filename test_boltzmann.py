import boltzmann_analysis
import matplotlib.pyplot as plt
import numpy as np

r = boltzmann_analysis.BoltzmannRun('/Users/conrad/repos/Boltzmann/output/')


# r.plot_spatial('number_luminosity_lab', 20)
# r.plot_spatial('energy_luminosity_lab', 20)
# r.plot_spectrum(20)
# r.plot_moments(20)
r.plot_flux_factors(20)


# radius = r[20].r
# volume = r[20].volume[:,0,0]
# area_r = r[20].area_r[1:,0,0]
# f = r[20].f[:,0,0,2,6,2]
#
# plt.plot(radius, volume / area_r / radius)

r.close()

plt.show()
