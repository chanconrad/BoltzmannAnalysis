import numpy as np
import h5py
from glob import glob
import os.path
import matplotlib.pyplot as plt

class Dump:
    def __init__(self, filename):
        self.filename = filename

        # Handle for dump file
        self.h5file = h5py.File(self.filename, mode = 'r')

        # Keys of all the data in the file
        self.keys = list(self.h5file.keys())

        for key in self.keys:
            setattr(self, key, self.value(key))

    def value(self, name):
        return np.array(self.h5file[name])


class BoltzmannDump(Dump):
    def number_density(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,None,:,:]
        eps = self.value('eps')[None,None,None,:,None,None]
        eps_if = self.value('eps_if')
        deps = (eps_if[1:] - eps_if[:-1])[None,None,None,:,None,None]
        j = np.sum(f * domega * eps**2 * deps, axis = (3,4,5))
        return j

    def zeroth_number_moment(self):
        return self.number_density() / (4.0 * np.pi)

    def energy_density(self):
        f = self.value('f')
        domega = self.value('domega')[None,None,None,None,:,:]
        eps = self.value('eps')[None,None,None,:,None,None]
        eps_if = self.value('eps_if')
        deps = (eps_if[1:] - eps_if[:-1])[None,None,None,:,None,None]
        j = np.sum(f * domega * eps**3 * deps, axis = (3,4,5))
        return j

    def zeroth_energy_moment(self):
        return self.energy_density() / (4.0 * np.pi)

    def energy_flux_density_radial(self):
        f = self.value('f')
        mu = self.value('mu')[None,None,None,None,:,None]
        domega = self.value('domega')[None,None,None,None,:,:]
        eps = self.value('eps')[None,None,None,:,None,None]
        eps_if = self.value('eps_if')
        deps = (eps_if[1:] - eps_if[:-1])[None,None,None,:,None,None]
        return np.sum(f * mu * domega * eps**3 * deps, axis = (3,4,5)) / (4.0 * np.pi)

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
        if direction==1:
            x = 'r'
            yval = dump.derived_value(y).mean(axis=(1,2))
        elif direction==2:
            x = 'theta'
            yval = dump.derived_value(y).mean(axis=(0,2))
        elif direction==3:
            x = 'phi'
            yval = dump.derived_value(y).mean(axis=(0,1))
        else:
            print('Invalid direction, must be 1, 2, or 3')

        xval = dump.value(x)

        plt.plot(xval, yval, label=f'{dump.time:4.2f}')

        plt.xlabel(x)
        plt.ylabel(y)

        plt.legend(title='time')


    def __getitem__(self, val):
        return self.dump[val]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'BoltzmannRun({self.run_path})'
