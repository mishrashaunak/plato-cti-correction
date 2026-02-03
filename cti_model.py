import numpy as np
from numba import prange, njit
import math

@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def run_cdm_parallel_cumul(
    image: np.ndarray,
    y0: int,
    beta: float,
    vg: float,
    t: float,
    fwc: float,
    vth: float,
    tr: np.ndarray,
    cnt: np.ndarray,
    sigma: np.ndarray,
    ci: bool = False
) -> np.ndarray:
    r"""
    Run `CDM` in the parallel direction.
    
    This version was adapted from the routine run_cdm_parallel() taken from Pyxel 1.10.2 
    (https://esa.gitlab.io/pyxel/).
    It accounts for the spatial variation of the traps across the CCD.
    
    Parameters
    ----------
    image: ndarray
        Input image.
    y0: int
        location of the first row in the CCD.
    beta: float
        Electron cloud expansion coefficient.
    vg: float
        Maximum geometrical volume.
    t: float
        Transfer period.
    fwc: float
        Full well capacity.
    vth: float
        Electron thermal velocity.
    tr: sequence of float
        Trap release time constants.
    cnt[4510,nx,ns] : cumulated number of traps along the column (i) up to row (j) for the species (k)
    sigma: sequence of float
        Trap capture cross section.
    ci: bool
        frame, the input frame assumed to contain CI and it is transferred into the CCD area.
        Default: False

    Returns
    -------
    array: ndarray
        Output array.
    """
    array = image.copy()  # we do a copy to preserve the input image
    ydim, xdim = array.shape  # full signal array we want to apply cdm for
    kdim_p = cnt.shape[2]

    # IMAGING (non-TDI) MODE
    # Parallel direction
    no = np.zeros((xdim, kdim_p))
    alpha_p = t * sigma * vth * fwc**beta / (2.0 * vg)
    g_p = 2.0 * cnt / fwc**beta
    
    for j in prange(xdim):
        for i in range(ydim):
            for k in range(kdim_p):
                nc = 0.0
                if array[i, j] > 0.01:
                    if (ci):
                        if (i + y0 < 4510):
                            gamma = g_p[4509, j, k] - g_p[i + y0, j, k]
                        else:
                            gamma = 0.
                    else:
                        if (i + y0 < 4510):
                            gamma = g_p[i + y0, j, k]
                        else:
                            gamma = g_p[4509, j, k]
                    
                    # Fixed syntax for multiline calculation
                    nc = max(
                        (gamma * array[i, j] ** beta - no[j, k])
                        / (gamma * array[i, j] ** (beta - 1.0) + 1.0)
                        * (1.0 - np.exp(-1 * alpha_p[k] * array[i, j] ** (1.0 - beta))),
                        0.0
                    )  # Eq. 22 in Short et al (2013)
                    no[j, k] += nc
                
                nr = no[j, k] * (1.0 - np.exp(-t / tr[k]))
                array[i, j] += -1 * nc + nr
                no[j, k] -= nr
                
                if array[i, j] < 0.01:
                    array[i, j] = 0.0

    return array

@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def run_cdm_parallel_cumul_radial_poly(
    image: np.ndarray,
    colindex: np.ndarray,
    y0: int,
    beta: float,
    vg: float,
    t: float,
    fwc: float,
    vth: float,
    tr: np.ndarray,
    a: np.ndarray,
    xc: float,
    yc: float,
    sigma: np.ndarray,
    ci: bool = False
) -> np.ndarray:
    r"""
    Run :term:`CDM` in parallel direction while taking account the trap distribution across the CCD.
    Here the trap density is modelled on the basis of a polynomial model 
    varying as a function of the distance from the centre of the focal plane.
    
    Adapted from run_cdm_parallel() taken from Pyxel 1.10 (https://esa.gitlab.io/pyxel/).

    Parameters
    ----------
    image: ndarray
        Input image.
    colindex: integer array
        index of the columns provided in the input image
    y0: int
        location of the first row in the CCD covered by input array.
    beta: float
        Electron cloud expansion coefficient.
    vg: float
        Maximum geometrical volume.
    t: float
        Transfer period.
    fwc: float
        Full well capacity.
    vth: float
        Electron thermal velocity.
    tr: sequence of float
        Trap release time constants.
    a: ndarray (ns,deg+1)
        Polynomial coefficients modelling the trap density as a function of the radius.
    xc,yc: (int,int)
        positions of the principal point (in the CCD reference frame)
    sigma: sequence of float
        Trap capture cross section.
    ci: bool
        frame, the input frame assumed to contain CI and it is transferred into the CCD area.
        Default: False

    Returns
    -------
    array: ndarray
        Output array.
    """
    array = image.copy()  # we do a copy to preserve the input image
    ydim, xdim = array.shape  # full signal array we want to apply cdm for
    kdim_p = a.shape[0]  # number of species

    # IMAGING (non-TDI) MODE
    # Parallel direction
    no = np.zeros((xdim, kdim_p))
    alpha_p = t * sigma * vth * fwc**beta / (2.0 * vg)

    # < derive the cumulated trap density (not computed for all the CCD rows)
    cnt = np.zeros((kdim_p, 4510, xdim))  # cumulated trap density
    Np = a.shape[1]
    D = np.zeros((4510, xdim, Np))
    
    for i in prange(y0, 4510):
        for j in range(xdim):
            RsqX = ((colindex[j] - xc) / 4510) ** 2 / 2.
            for p in range(Np):
                for ip in range(i + 1):
                    # Added import math for this line
                    R = math.sqrt(RsqX + ((ip - yc) / 4510) ** 2 / 2.)
                    D[i, j, p] += R**p
                    
    for i in prange(y0, 4510):
        for k in range(kdim_p):
            for j in range(xdim):
                for p in range(Np):
                    cnt[k, i, j] += a[k, p] * D[i, j, p]
    # >

    g_p = 2.0 * cnt / fwc**beta
    
    for j in prange(xdim):
        for i in range(ydim):
            for k in range(kdim_p):
                nc = 0.0
                if array[i, j] > 0.01:
                    if (ci):
                        if (i + y0 < 4510):
                            gamma = g_p[k, 4509, j] - g_p[k, i + y0, j]
                        else:
                            gamma = 0.
                    else:
                        if (i + y0 < 4510):
                            gamma = g_p[k, i + y0, j]
                        else:
                            gamma = g_p[k, 4509, j]
                    
                    # Fixed syntax for multiline calculation
                    nc = max(
                        (gamma * array[i, j] ** beta - no[j, k])
                        / (gamma * array[i, j] ** (beta - 1.0) + 1.0)
                        * (1.0 - np.exp(-1 * alpha_p[k] * array[i, j] ** (1.0 - beta))),
                        0.0
                    )  # Eq. 22 in Short et al (2013)
                    no[j, k] += nc
                
                nr = no[j, k] * (1.0 - np.exp(-t / tr[k]))
                array[i, j] += -1 * nc + nr
                no[j, k] -= nr
                
                if array[i, j] < 0.01:
                    array[i, j] = 0.0

    return array
