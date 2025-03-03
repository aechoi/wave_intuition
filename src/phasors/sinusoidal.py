from dataclasses import dataclass
import numpy as np


@dataclass
class SinusoidSignal:
    """A representation of a sinusoidal signal as a function of space and time. Both
    the phasor domain and time domain are stored.

    The function is defined on a domain of space and domain of time.

    Needs to be able to plot
        - v(z, t) = Re{p}
        - the phasor ellipse p
        - particular points in space in the time domain within a particular period (tiled)
        - particular points in space in the phasor domain within one rotation

    """

    _magnitude: float
    _phase: float
    beta: float
    omega: float
    max_time: float = 10
    max_space: float = 1

    def __post_init__(self):
        self.time = np.linspace(0, self.max_time, 1000)
        self._current_time = 0
        self._current_time_idx = 0
        self.space = np.linspace(0, self.max_space, 1000)
        self._current_loc = 0
        self.current_loc_idx = 0

        self.space_sample_indices = self.generate_z_samples()

        self.phasor = None  # complex number denoting phasor without time
        self.space_time_vec = None  # complex number of current phasor at time
        self.vzt = None  # v(z,t)

        self.set_phasor()
        self.current_phasor = self.space_time_vec[self._current_time_idx]

    def generate_z_samples(self, num_samples=8):
        if self.beta > 0:
            wavelength = 2 * np.pi / self.beta
            z_samples = np.linspace(0, wavelength, num_samples)
        else:
            z_samples = np.linspace(0, self.max_space, 8)
        sample_indices = np.argmin(
            np.abs(self.space[:, None] - z_samples[None, :]), axis=0
        )
        return sample_indices

    def set_phasor(self):
        """Recalculates the spatial components of the phasor"""
        self.phasor = (
            self._magnitude
            * np.exp(self._phase * 1j)
            * np.exp(-self.beta * self.space * 1j)
        )  # dim = space
        self.space_time_vec = (
            self.phasor[None, :] * np.exp(self.omega * self.time * 1j)[:, None]
        )  # dim = time x space
        self.vzt = np.real(self.space_time_vec)  # dim = time x space

    def get_time_index(self, time):
        return np.argmin(np.abs(self.time - time))

    def get_space_index(self, loc):
        return np.argmin(np.abs(self.space - loc))

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, val):
        self._magnitude = val
        self.set_phasor()
        self.current_phasor = self.space_time_vec[self._current_time_idx]

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, val):
        self._phase = val
        self.set_phasor()
        self.current_phasor = self.space_time_vec[self._current_time_idx]

    @property
    def current_time(self):
        return self._current_time

    @current_time.setter
    def current_time(self, val):
        self._current_time = val
        self._current_time_idx = self.get_time_index(val)
        self.current_phasor = self.space_time_vec[self._current_time_idx]

    @property
    def current_loc(self):
        return self._current_loc

    @current_loc.setter
    def current_loc(self, val):
        self._current_loc = val
        self.current_loc_idx = self.get_space_index(val)
