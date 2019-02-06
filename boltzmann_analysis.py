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
        r_if = self.value('r_if')[:,None,None]
        area_r = self.value('area_r')
        # Cross section (aera_r) at the center of the cell
        cs = 0.25 * area_r[:-1,:,:] + 0.25 * area_r[1:,:,:] + 0.5 * area_r[1:,:,:] * r_if[:-1,:,:] / r_if[1:,:,:]
        return cs

    def epsvol(self):
        # Note: eps**2 * deps = (4/3) * (eps_if[1]**3 - eps_if[0]**3)
        eps_if = self.value('eps_if')
        return ( (4.0/3.0) * (eps_if[1:]**3 - eps_if[:-1]**3) )

    def number_density(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,None,:,:]
        epsvol = self.epsvol()[None,None,None,:,None,None]

        return np.sum(f * domega * epsvol, axis = (3,4,5))

    def zeroth_number_moment(self):
        return self.number_density() / (4.0 * np.pi)

    def first_number_moment(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,None,:,:]
        epsvol = self.epsvol()[None,None,None,:,None,None]
        mu = self.value('mu')[None,None,None,None,:,None]
        return np.sum(f * domega * epsvol * mu, axis = (3,4,5)) / (4.0 * np.pi)

    def second_number_moment(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,None,:,:]
        epsvol = self.epsvol()[None,None,None,:,None,None]
        mu = self.value('mu')[None,None,None,None,:,None]
        return np.sum(f * domega * epsvol * mu**2, axis = (3,4,5)) / (4.0 * np.pi)

    def energy_density(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,None,:,:]
        eps = self.value('eps')[None,None,None,:,None,None]
        epsvol = self.epsvol()[None,None,None,:,None,None]
        j = np.sum(f * domega * eps * epsvol, axis = (3,4,5))
        return j

    def zeroth_energy_moment(self):
        return self.energy_density() / (4.0 * np.pi)

    def number_flux_density_radial(self):
        f = self.value('f')
        mu = self.value('mu')[None,None,None,None,:,None]
        domega = self.value('domega')[None,None,None,None,:,:]
        epsvol = self.epsvol()[None,None,None,:,None,None]
        return np.sum(f * mu * domega * epsvol, axis = (3,4,5)) / (4.0 * np.pi)

    def energy_flux_density_radial(self):
        f = self.value('f')
        mu = self.value('mu')[None,None,None,None,:,None]
        domega = self.value('domega')[None,None,None,None,:,:]
        eps = self.value('eps')[None,None,None,:,None,None]
        epsvol = self.epsvol()[None,None,None,:,None,None]
        return np.sum(f * mu * domega * eps * epsvol, axis = (3,4,5)) / (4.0 * np.pi)

    def energy_luminosity(self):
        # return 16.0 * np.pi**2 * r**2 * self.energy_flux_density_radial()
        return 4.0 * np.pi * self.cross_section_r() * self.energy_flux_density_radial()

    def energy_luminosity_lab(self):
        vfluid_r = self.value('vfluid')[:,:,:,0]
        return self.energy_luminosity() * (1.0 + vfluid_r) / (1.0 - vfluid_r)

    def number_luminosity(self):
        r = self.value('r')[:,None,None]
        r_if = self.value('r_if')[:,None,None]
        # return 16.0 * np.pi**2 * r**2 * self.number_flux_density_radial()
        return 4.0 * np.pi * self.cross_section_r() * self.number_flux_density_radial()

    def number_luminosity_lab(self):
        vfluid_r = self.value('vfluid')[:,:,:,0]
        return self.number_luminosity() * self.gamma() * (1.0 + vfluid_r)

    def gamma(self):
        vfluid = self.value('vfluid')
        clight = float(self.value('clight'))
        beta2 = np.sum((vfluid / clight)**2, axis=3)
        gamma = 1.0 / np.sqrt(1.0 - beta2)
        return gamma

    def spectrum(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,None,:,:]

        return np.sum(f * domega, axis = (4,5)) / (4.0 * np.pi)

    def radiating_sphere_exact_solution(self):
        i_sphere = self.value('kappa_a').max(axis=3).argmin()-1
        r_sphere = self.value('r')[i_sphere]
        kappa_a_sphere = self.value('kappa_a').max()
        f_eq_sphere = self.value('f_eq').max()

        eps_if = self.value('eps_if')
        epsvol = ( (4.0/3.0) * (eps_if[1:]**3 - eps_if[:-1]**3) )
        i_energy = np.argmax(self.value('kappa_a')[0,0,0,:])
        f_eq_sphere *= epsvol[i_energy]

        sol = RadiatingSphere(r_sphere, kappa_a_sphere, f_eq_sphere)

        moments = np.zeros((len(self.value('r')), 3))

        for i, r in enumerate(self.value('r')):
            j, h, k = sol.moments(r)
            moments[i,0] = j
            moments[i,1] = h
            moments[i,2] = k

        return moments

    def derived_value(self, name):
        return getattr(self, name)()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'BoltzmannDump({self.filename})'

class BoltzmannRun:
    def __init__(self, run_path, file_extension='h5'):
        self.run_path = run_path
        self.file_extension = file_extension

        # Look for dump files in directory
        search_path = os.path.join(self.run_path, f'*.{self.file_extension}')
        self.dump_files = sorted(glob(search_path))

        self.load_dumps()

    def load_dumps(self):
        '''Generate the index for dump files'''
        self.dump = []
        for i, file in enumerate(self.dump_files):
            dump = BoltzmannDump(file)
            ind = int(dump.value('index'))
            assert ind == i, 'Files out of order'
            self.dump += [dump]

    def plot_spatial(self, y, index, direction=1):
        '''Plot y against spatial coordinate'''

        dump = self.dump[index]
        yval, x = self._select_axis(dump.derived_value(y), direction)

        xval = dump.value(x)

        plt.plot(xval, yval, label=f'{dump.time:4.2f}')

        plt.xlabel(x)
        plt.ylabel(y)

        plt.legend(title='time')

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

    def plot_mu_distribution(self, index):
        '''Plot each of the mu bins'''
        dump = self.dump[index]

        f = dump.f
        nmu = f.shape[4]

        f_av = np.mean(f, axis=(1,2,3,5))

        for i in range(nmu):
            plt.plot(f_av[:,i], label = f'{dump.mu[i]:0.2f}')

        plt.xlabel('r')
        plt.ylabel('f')

        plt.legend()

    def plot_moments(self, index, show_exact=True):
        '''Plot the moments, with an option to compare with exact solution for radiating sphere'''
        dump = self.dump[index]

        j, x = self._select_axis(dump.zeroth_number_moment(), 1)
        h, x = self._select_axis(dump.first_number_moment(), 1)
        k, x = self._select_axis(dump.second_number_moment(), 1)
        plt.plot(dump.value('r'), j)
        plt.plot(dump.value('r'), h)
        plt.plot(dump.value('r'), k)

        if show_exact:
            exact_moments = dump.radiating_sphere_exact_solution()
            plt.plot(dump.value('r'), exact_moments[:,0], ls='--', label = 'J')
            plt.plot(dump.value('r'), exact_moments[:,1], ls='--', label = 'H')
            plt.plot(dump.value('r'), exact_moments[:,2], ls='--', label = 'K')

        plt.xlabel('r')
        plt.ylabel('J,H,K')

        plt.legend()

    def plot_flux_factors(self, index, show_exact=True):
        '''Plot the flux factors, with an option to compare with exact solution for radiating sphere'''
        dump = self.dump[index]

        j, x = self._select_axis(dump.zeroth_number_moment(), 1)
        h, x = self._select_axis(dump.first_number_moment(), 1)
        k, x = self._select_axis(dump.second_number_moment(), 1)
        plt.plot(dump.value('r'), h/j)
        plt.plot(dump.value('r'), k/j)

        if show_exact:
            exact_moments = dump.radiating_sphere_exact_solution()
            plt.plot(dump.value('r'), exact_moments[:,1]/exact_moments[:,0], ls='--', label = 'H/J')
            plt.plot(dump.value('r'), exact_moments[:,2]/exact_moments[:,0], ls='--', label = 'K/J')

        plt.xlabel('r')
        plt.ylabel('H/J, K/J')

        plt.legend()

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
