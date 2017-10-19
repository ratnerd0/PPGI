
import abc
import multiprocessing
_NCORES = multiprocessing.cpu_count()

import numpy as np
from scipy import ndimage

from matplotlib import pyplot as plt


try:
    import pyfftw
    np.fft = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(15.0)
except ImportError as e:
    print "could not use pyFFTW"
    
    
    
import fftw3

"""
Code for performing phase retrieval.

The classic phase retrieval problem is, given a set of modulus measurements
of a complex function f, ie. given F

    F = | f |

recover the phases of f. This is possible if f is 2 or greater dimensional,
and F is 2x or greater oversampled.
"""


def low_res_cut(e_map, factor):
    """
    Factor is the fraction of voxels retained in each dimension.
    """
    
    shp = np.array(e_map.shape)
    cut = e_map.copy()

    center = (shp/2).astype(np.int)
    width  = (shp * (factor) / 2).astype(np.int)
    
    l_slice = center - width
    r_slice = center + width + 1
    
    l_slice = l_slice.astype(np.int)
    r_slice = r_slice.astype(np.int)
    
    cut = cut[l_slice[0]:r_slice[0],
              l_slice[1]:r_slice[1],
              l_slice[2]:r_slice[2]]
    
    return cut

def expand_support(support, new_shape):
    """
    Pad with zeros, then grow 
    """
    zoom_factor = np.array(new_shape).astype(np.float) / \
                    np.array(support.shape).astype(np.float)
    expd_support = ndimage.interpolation.zoom(support, zoom_factor, order=0)
    assert expd_support.shape == tuple(new_shape), str(expd_support.shape) + str(new_shape)
    return expd_support


def expand_guess(guess, new_shape):
    
    # FT, pad with zeros, invFT
    ft_guess = np.fft.fftn(guess)
    
    pad_width = (np.array(new_shape) - np.array(guess.shape)) / 2
    if np.any(pad_width % 1 != 0.0):
        raise ValueError('New shape must enable even padding on both sides')
    
    pad_width = [(int(p),int(p)) for p in pad_width]
    
    ft_guess = np.pad(ft_guess, pad_width, 
                      mode='constant', constant_values=0)
    expd_guess = np.fft.ifftn(ft_guess)
    
    assert expd_guess.shape == tuple(new_shape), str(expd_guess.shape) + str(new_shape)
    
    return expd_guess


def init_support(e_map):
    
    acf = np.fft.fftshift(np.abs(np.fft.ifftn(e_map)))

    s = (acf > acf.max() * 0.01)
    s = ndimage.morphology.binary_fill_holes(s)
    s = ndimage.morphology.binary_dilation(s, iterations=4)
    
    return s


class IterativePhaser(object):
    """
    A base class for phase iterative phase retrieval algorithms. Appropriate
    for any method that can (and should) be written in terms of alternating
    projections on two sets.
    """
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,
                 enable_shrinkwrap=True,
                 shrinkwrap_freq=15,
                 shrinkwrap_threshold=0.04, 
                 shrinkwrap_sigma=2.0,
                 inflate=0,
                 fill_holes=True,
                 max_iterations=500, 
                 conv_tol=1e-6,
                 plot=False,
                 verbose=False):
        """
        Parameters
        ----------         
         enable_shrinkwrap : bool
             Enable the shrinkwrap algorithm, which allows for support estimate. 
             You probably either have to do this or provide a good estimate of 
             the support.
         
         shrinkwrap_freq : int
             How many phase retrival iterations to perform before updating the
             support with shrinkwrap.
         
         shrinkwrap_threshold : float
             Fraction of the maximum intensity to set the support: e.g. if this
             value is 0.05, shrinkwrap will set all values less than 5% of the
             max intensity to be outside the support.
         
         shrinkwrap_sigma : float
             The size of the Gaussian blurring shrinkwrap will use. Set to 0 to
             turn this off
         
         fill_holes : bool
             Fill holes in the support estimate. Recommended if you know
             your object is continuous.
         
         max_iterations : int
             Max number of iterations to perform.
         
         conv_tol : float
             An error tolerance for algorithm termination.
        """
                 
        self.shrinkwrap_on        = enable_shrinkwrap
        self.shrinkwrap_threshold = shrinkwrap_threshold
        self.shrinkwrap_sigma     = shrinkwrap_sigma
        self.shrinkwrap_freq      = shrinkwrap_freq
        self.inflate              = inflate
        self.fill                 = fill_holes
        self.max_iterations       = max_iterations
        self.conv_tol             = conv_tol
        self.plot                 = plot
        self.verbose              = verbose
        
        self.support = None
        self.errors  = []
        
        return
        
        
    def __call__(self, fourier_map, guess=None, mask=None):
        """
        Run the phase retrieval algorithm on a Fourier measurement,
        `fourier_map`.
        
        Parameters
        ----------
        fourier_map : np.ndarray
            The reciprocal-space (Fourier) intensities. NOTE this value will
            is assumed to be the squared amplitudes... it makes a difference!
    
        guess : bool
            An intial real-space image to start with. Can be an informative
            guess, or you can use this argument to put two methods in
            serial, e.g. you might want to do Error Reduction after HIO.
                 
        mask : np.ndarray (bool)
            A mask of values in fourier_map to ignore (False) and keep (True).
            If `mask` is `None`, then all measurements are kept.
    
        Returns
        -------
        self.image : np.ndarray
            The phase-recovered real-space image.
        """
                 
        assert np.all(fourier_map >= 0.0)
    
        self.fourier_map = fourier_map
        self._sqI = np.sqrt(self.fourier_map)
        
        if self.support is None:
            self.support = self._acf_support(self.fourier_map)
        
        if mask is None:
            self.mask = np.s_[:]
        else:
            self.mask = mask
        
        
        if guess is None:
            self.image = np.random.randn(*fourier_map.shape) + \
                            np.zeros(fourier_map.shape) * 1j
            self.image[ np.logical_not(self.support) ] = 0.0 + 0.0 * 1j
        else:
            self.image = guess
        self.previous_image = self.image.copy()
    
        # iterate
        self.errors = []
        for i in range(self.max_iterations):
            
            self._iteration = i # for monkeypatching
            
            err = self._project_inv()
            self._project_real()
            self.image = self.realspace_constraints(self.image)
            self.errors.append(err)
            if self.verbose:
                print 'Error: %.2e' % err
            
            if ((i % self.shrinkwrap_freq) == 0) and self.shrinkwrap_on and (i>0):
                self.shrinkwrap()
            self.support = self.support_update(self.support)
            
            if self.plot:
                self._plot()
            
            if err < self.conv_tol:
                if self.verbose:
                    print "convergence tolerance reached on iteration %d" % (i+1)
                break
    
        return self.image
        
        
    def _acf_support(self, fourier_map):
        """
        Estimate the initial support using the autocorrelation function.
        """
        acf = np.abs( np.fft.ifftn(fourier_map) )
        support = (acf > (np.max(acf) * self.shrinkwrap_threshold * 0.9))
        support = np.fft.fftshift(support)
        return support
        
        
    def shrinkwrap(self):
        """
        Perform an interation of the shrinkwrap algorithm, which provides
        an estimate of the object support.
        """
        
        before = np.sum(self.support)
        
        if self.shrinkwrap_sigma > 0.0:
            im = ndimage.filters.gaussian_filter(self.image.real, 
                                                 self.shrinkwrap_sigma,
                                                 mode='constant')
            if self.shrinkwrap_sigma > 1.5:
                self.shrinkwrap_sigma *= 0.99
        else:
            im = self.image.real
            
        t = self.shrinkwrap_threshold * np.max(im)
        self.support = (im > t) * self.support
            
        if self.inflate > 0:
            self.support = ndimage.morphology.binary_dilation(self.support, 
                                                              iterations=self.inflate)
                                                              
        
        if self.fill:
            self.support = ndimage.binary_fill_holes(self.support)
        
        after = np.sum(self.support)
        print "shrinkwrap'd support %d --> %d voxels" % (before, after)
        
        return
        
        
    def _plot(self):
        
        if not hasattr(self, '_fig'):
            plt.ion()
            print 'plotting'
            self._fig = plt.figure()
            self._ax1 = plt.subplot(121)
            self._ax2 = plt.subplot(122)
        
        self._ax1.imshow(self.image.real, interpolation='nearest')
        self._ax2.imshow(self.support, interpolation='nearest', cmap=plt.cmap.grey)
        plt.pause(1e-16)
        
        return
    
        
    @abc.abstractmethod
    def _project_inv(self):
        """
        Project on the inverse-space (Fourier space) set
        """
        pass
        
        
    @abc.abstractmethod
    def _project_real(self):
        """
        Project on the real-space set
        """
        pass
        
    def realspace_constraints(self, image):
        # placeholder for monkeypatching
        return image
        
    def support_update(self, support):
        # placeholder for monkeypatching
        return support
        
    
class ErrorReduction(IterativePhaser):
    
    def _project_inv(self):
        
        inv = np.fft.fftn(self.image, overwrite_input=False, planner_effort='FFTW_MEASURE', threads=_NCORES)
        inv_r   = np.abs(inv)
        inv_phi = np.angle(inv)
        
        error = np.linalg.norm(inv_r - self._sqI)
        inv_r[self.mask] = self._sqI[self.mask].copy()
        
        self.image = np.fft.ifftn( inv_r * np.exp(1j * inv_phi), overwrite_input=False, planner_effort='FFTW_MEASURE', threads=_NCORES)
        
        return error
        
    def _project_real(self):
        self.image[ np.logical_not(self.support) ] = 0.0 + 0.0j
        self.image.imag = 0.0
        #self.image[ self.image.real < 0.0 ] = 0.0 + 0.0j
        return
        
        
        
class HIO(ErrorReduction):
    """
    HIO is the most commonly employed phase retrieval algorithm. It is simple,
    fast, and generally performs well.
    
    The out-of-support update is:
        Dx = x - Bx'
    
    Otherwise the algorithm is the same as Error Reduction.
    """
    
    beta = 0.9
        
    def _project_real(self):
        
        # HIO support update
        ns = np.logical_not(self.support)
        self.image[ns] = self.previous_image[ns] - self.beta * self.image[ns]
        
        # reality
        self.image.imag = 0.0
        
        # positivity
        #self.image[ self.image.real < 0.0 ] = 0.0 + 0.0j
        
        # update
        self.previous_image = self.image.copy()
        
        return
        
        
class RAAR(ErrorReduction):
    """
    For RAAR, the out-of-support update is
    
        Dx = Bx + (1 - 2B)(x')
    """
    
    beta = 0.75

    def _project_real(self):
        
        # RAAR support update
        ns = np.logical_not(self.support)
        self.image[ns] = self.beta * self.previous_image[ns] + \
                            (1.0 - 2.0 * self.beta) * self.image[ns]
        
        # reality
        self.image.imag = 0.0
        
        # update
        self.previous_image = self.image.copy()
        
        return
        
