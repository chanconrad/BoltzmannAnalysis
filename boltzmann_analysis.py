import numpy as np
import h5py
from glob import glob
import os.path
import matplotlib.pyplot as plt
from exact_solutions import RadiatingSphere

class Dump:
    def __init__(self, filename):
        self.filename = filename

        self.open_file()

        # Keys of all the data in the file
        self.keys = list(self.h5file.keys())

        for key in self.keys:
            setattr(self, key, self.value(key))


    def open_file(self):
        self.h5file = h5py.File(self.filename, mode = 'r')

    def close_file(self):
        self.h5file.close()

    def value(self, name):
        return np.array(self.h5file[name])


class BoltzmannDump(Dump):
    def cross_section_r(self):
        # r_if = self.value('r_if')[:,None,None]
        # area_r = self.value('area_r')
        # Cross section (area_r) at the center of the cell
        # cs = 0.25 * area_r[:-1,:,:] + 0.25 * area_r[1:,:,:] + 0.5 * area_r[1:,:,:] * r_if[:-1,:,:] / r_if[1:,:,:]

        cs = 4.0*np.pi*self.value('r')[:,None,None,None]**2

        return cs

    def epsvol(self):
        # Note: eps**2 * deps = (4/3) * (eps_if[1]**3 - eps_if[0]**3)
        eps_if = self.value('eps_if')
        # return ( (4.0/3.0) * (eps_if[1:]**3 - eps_if[:-1]**3) )
        return ( (1.0/3.0) * (eps_if[1:]**3 - eps_if[:-1]**3) )

    def number_density(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,:,:,None,None]
        epsvol = self.epsvol()[None,None,None,None,None,:,None]

        return np.sum(f * domega * epsvol, axis = (3,4,5))

    def zeroth_number_moment(self):
        return self.number_density() / (4.0 * np.pi)

    def first_number_moment(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,:,:,None,None]
        epsvol = self.epsvol()[None,None,None,None,None,:,None]
        mu = self.value('mu')[None,None,None,:,None,None,None]
        return np.sum(f * domega * epsvol * mu, axis = (3,4,5)) / (4.0 * np.pi)

    def second_number_moment(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,:,:,None,None]
        epsvol = self.epsvol()[None,None,None,None,None,:,None]
        mu = self.value('mu')[None,None,None,:,None,None,None]
        return np.sum(f * domega * epsvol * mu**2, axis = (3,4,5)) / (4.0 * np.pi)

    def energy_density(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,:,:,None,None]
        eps = self.value('eps')[None,None,None,None,None,:,None]
        epsvol = self.epsvol()[None,None,None,None,None,:,None]
        mu = self.value('mu')[None,None,None,:,None,None,None]
        j = np.sum(f * domega * eps * epsvol, axis = (3,4,5))
        return j

    def zeroth_energy_moment(self):
        return self.energy_density() / (4.0 * np.pi)

    def number_flux_density_radial(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,:,:,None,None]
        epsvol = self.epsvol()[None,None,None,None,None,:,None]
        mu = self.value('mu')[None,None,None,:,None,None,None]
        return np.sum(f * mu * domega * epsvol, axis = (3,4,5)) / (4.0 * np.pi)

    def energy_flux_density_radial(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,:,:,None,None]
        eps = self.value('eps')[None,None,None,None,None,:,None]
        epsvol = self.epsvol()[None,None,None,None,None,:,None]
        mu = self.value('mu')[None,None,None,:,None,None,None]
        return np.sum(f * mu * domega * eps * epsvol, axis = (3,4,5)) / (4.0 * np.pi)

    def energy_luminosity(self):
        # return 16.0 * np.pi**2 * r**2 * self.energy_flux_density_radial()
        return 4.0 * np.pi * self.cross_section_r() * self.energy_flux_density_radial()

    def energy_dldr(self):
        l = self.energy_luminosity()[:,0,0,:]
        r = self.value('r')

        dldr = np.zeros_like(l)

        dldr[1:-1,:] = (l[2:,:] - l[:-2,:]) / (r[2:] - r[:-2])[:,None]
        dldr[0,:] = (l[1,:] - l[0,:]) / (r[1] - r[0])[None]
        dldr[-1,:] = (l[-1,:] - l[-2,:]) / (r[-1] - r[-2])[None]

        return dldr[:,None,None,:]

    def energy_luminosity_lab(self):
        vfluid_r = self.value('vfluid')[:,:,:,0]
        factor = (1.0 + vfluid_r) / (1.0 - vfluid_r)
        return self.energy_luminosity() * factor[:,None]

    def number_luminosity(self):
        # return 16.0 * np.pi**2 * r**2 * self.number_flux_density_radial()
        return 4.0 * np.pi * self.cross_section_r() * self.number_flux_density_radial()

    def number_luminosity_lab(self):
        vfluid_r = self.value('vfluid')[:,:,:,0]
        factor = self.gamma() * (1.0 + vfluid_r)

        return self.number_luminosity() * factor[:,None]

    def gamma(self):
        vfluid = self.value('vfluid')
        clight = float(self.value('clight'))
        beta2 = np.sum((vfluid / clight)**2, axis=3)
        gamma = 1.0 / np.sqrt(1.0 - beta2)
        return gamma

    def spectrum(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,:,:,None,None]

        return np.sum(f * domega, axis = (3,4)) / (4.0 * np.pi)

    def radiating_sphere_exact_solution(self):
        kappa_a = self.value('kappa_a')
        f_eq = self.value('f_eq')

        eps_if = self.value('eps_if')
        epsvol = self.epsvol()

        r = self.value('r')

        ngrid = 4 * len(r)
        rgrid = np.linspace(r[0], r[-1], ngrid)

        moments = np.zeros((self.nflav, ngrid, 3))

        for l in range(self.nflav):
            i_sphere = kappa_a[...,l].max(axis=3).argmin()-1
            r_sphere = self.value('r')[i_sphere]
            kappa_a_sphere = kappa_a[...,l].max()
            f_eq_sphere = f_eq[...,l].max()

            i_energy = np.argmax(kappa_a[0,0,0,:,l])
            f_eq_sphere *= epsvol[i_energy]

            sol = RadiatingSphere(r_sphere, kappa_a_sphere, f_eq_sphere)

            for i, r in enumerate(rgrid):
                j, h, k = sol.moments(r)
                moments[l,i,0] = j
                moments[l,i,1] = h
                moments[l,i,2] = k

        return rgrid, moments

    def derived_value(self, name):
        value = getattr(self, name)
        if type(value) is np.ndarray:
            return value
        else:
            return value()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'BoltzmannDump({self.filename})'

class BoltzmannRun:
    def __init__(self, run_path, file_extension='h5', search_pattern=None):
        self.run_path = run_path
        self.file_extension = file_extension

        # Look for dump files in directory
        if search_pattern is None:
            search_path = os.path.join(self.run_path, f'*.{self.file_extension}')
        else:
            search_path = os.path.join(self.run_path, search_pattern)
        self.dump_files = sorted(glob(search_path))

        self.load_dumps()

    def load_dumps(self):
        '''Generate the index for dump files'''
        self.dump = []
        current_index = -1
        for i, file in enumerate(self.dump_files):
            dump = BoltzmannDump(file)
            ind = int(dump.value('index'))
            assert ind > current_index, 'Files out of order'
            current_index = ind
            dump.nflav = dump.value('f').shape[6]
            self.dump += [dump]

    def plot_spatial(self, y, index, flavour=0, direction=1, energy=0, logx=False, logy=False, **kwargs):
        '''Plot y against spatial coordinate'''

        dump = self.dump[index]
        yval, x = self._select_axis(dump.derived_value(y), direction)

        xval = dump.value(x)

        if (len(yval.shape) == 3):
            yval = yval[:,energy,flavour]
        elif (len(yval.shape) == 2):
            yval = yval[:,flavour]

        if 'label' in kwargs:
            label = kwargs.pop('label')
        else:
            label = f't={dump.time:4.2f}'

        plt.plot(xval, yval, label=label, **kwargs)

        plt.xlabel(x)
        plt.ylabel(y)

        plt.legend(title='time')

        if logx:
            plt.xscale('log')

        if logy:
            plt.yscale('log')

    def plot_spectrum(self, index, direction=1):
        '''Plot spectrum against spatial coordinate'''

        dump = self.dump[index]
        yval, x = self._select_axis(dump.spectrum(), direction)

        xval = dump.value(x)
        eps = dump.eps

        for i, e in enumerate(eps):
            plt.plot(xval, yval[:,i], label=f'{e:4.2f}')

        plt.legend(title='energy')
        plt.xlabel(x)
        plt.ylabel('f')

    def plot_mu_distribution(self, index, energy=0, flavour=0, logx=False, logy=False):
        '''Plot each of the mu bins'''
        dump = self.dump[index]

        f = dump.f
        nmu = f.shape[3]
        f_av = np.mean(f, axis=(1,2,4))
        r = dump.value('r').squeeze()
        r_if = dump.value('r_if').squeeze()

        i = energy
        l = flavour

        for j in range(nmu):
            plt.plot(r, f_av[:,j,i,l], label = f'{dump.mu[j]:0.2f}')

        plt.xlabel('r')
        plt.ylabel('f')

        plt.legend()

        if logx:
            plt.xscale('log')

        if logy:
            plt.yscale('log')

    def plot_moments(self, index, flavour=0, show_exact=False, logx=False, logy=False):
        '''Plot the moments, with an option to compare with exact solution for radiating sphere'''
        dump = self.dump[index]

        j, x = self._select_axis(dump.zeroth_number_moment(), 1)
        h, x = self._select_axis(dump.first_number_moment(),  1)
        k, x = self._select_axis(dump.second_number_moment(), 1)

        if show_exact:
            rgrid, exact_moments = dump.radiating_sphere_exact_solution()

        l = flavour

        plt.figure()

        plt.plot(dump.value('r'), j[:,l], label = 'J')
        plt.plot(dump.value('r'), h[:,l], label = 'H')
        plt.plot(dump.value('r'), k[:,l], label = 'K')

        if show_exact:
            plt.plot(rgrid, exact_moments[l,:,0], ls='--')
            plt.plot(rgrid, exact_moments[l,:,1], ls='--')
            plt.plot(rgrid, exact_moments[l,:,2], ls='--')

        plt.xlabel('r')
        plt.ylabel('J,H,K')

        plt.title(f't = {dump.time:0.2e}, flavour {l}')

        plt.legend()

        if logx:
            plt.xscale('log')

        if logy:
            plt.yscale('log')

    def plot_flux_factors(self, index, flavour=0, show_exact=False, logx=False, logy=False, **kwargs):
        '''Plot the flux factors, with an option to compare with exact solution for radiating sphere'''
        dump = self.dump[index]

        j, x = self._select_axis(dump.zeroth_number_moment(), 1)
        h, x = self._select_axis(dump.first_number_moment(),  1)
        k, x = self._select_axis(dump.second_number_moment(), 1)

        if show_exact:
            rgrid, exact_moments = dump.radiating_sphere_exact_solution()

        l = flavour
        # plt.figure()

        if 'label' in kwargs:
            label = kwargs.pop('label')
        else:
            label = ''

        plt.plot(dump.value('r'), (h/j)[:,l], label = label + ' H/J')
        plt.plot(dump.value('r'), (k/j)[:,l], label = label + ' K/J')

        if show_exact:
            plt.plot(rgrid, exact_moments[l,:,1]/exact_moments[l,:,0], ls='--')
            plt.plot(rgrid, exact_moments[l,:,2]/exact_moments[l,:,0], ls='--')

        plt.xlabel('r')
        plt.ylabel('H/J, K/J')

        plt.title(f't = {dump.time:0.2e}, flavour {l}')

        plt.legend()

        if logx:
            plt.xscale('log')

        if logy:
            plt.yscale('log')

    @staticmethod
    def _select_axis(vals, direction):
        if direction==1:
            x = 'r'
            yval = vals.mean(axis=(1,2))
        elif direction==2:
            x = 'theta'
            yval = vals.mean(axis=(0,2))
        elif direction==3:
            x = 'phi'
            yval = vals.mean(axis=(0,1))
        else:
            print('Invalid direction, must be 1, 2, or 3')

        return yval, x

    def open(self):
        for dump in self.dump:
            dump.open_file()

    def close(self):
        for dump in self.dump:
            dump.close_file()


    def __getitem__(self, val):
        return self.dump[val]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'BoltzmannRun({self.run_path})'
