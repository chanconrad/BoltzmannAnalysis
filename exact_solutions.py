'''
Exact solutions
'''

import numpy as np

class RadiatingSphere:
    def __init__(self, r_sphere, kappa_a_sphere, f_eq_sphere, nmu=1000):
        self.r_sphere = r_sphere
        self.kappa_a_sphere = kappa_a_sphere
        self.f_eq_sphere = f_eq_sphere
        self.nmu = nmu

    def moments(self, r):
        mu_range = np.linspace(-1, 1, self.nmu)

        j = 0.0
        h = 0.0
        k = 0.0

        for i in range(self.nmu-1):
            intensity = self.intensity(r, mu_range[i])
            dmu = 2.0 / float(self.nmu)
            st_l = np.sqrt(1.0 - mu_range[i]**2)
            st_r = np.sqrt(1.0 - mu_range[i+1]**2)
            st = 0.5*(st_l+st_r)
            ds = np.sqrt((st_l-st_r)**2 + dmu**2)
            domega = st * ds
            j += intensity * domega
            h += intensity * domega * mu_range[i]
            k += intensity * domega * mu_range[i]**2

        j /= 2.0
        h /= 2.0
        k /= 2.0

        return j, h, k

    def intensity(self, r, mu):
        if r < self.r_sphere:
            s = r * mu + self.r_sphere * self.g(r, mu)
        else:
            x = np.sqrt(1.0 - (self.r_sphere/r)**2)
            if x <= mu:
                s = 2.0 * self.r_sphere * self.g(r, mu)
            else:
                s = 0.0

        return self.f_eq_sphere * (1.0 - np.exp(-self.kappa_a_sphere * s))

    def g(self, r, mu):
        return np.sqrt(1.0 - (r/self.r_sphere)**2 * (1.0-mu**2))
