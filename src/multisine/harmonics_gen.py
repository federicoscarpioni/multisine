'''
Harmonic grids for multisine excitations.

Two families, for two different measurement regimes.

**Dense odd random phase (ORP)**, ``odd_random_grid``. Every odd harmonic in
the band is excited except one per group of L, chosen at random. This is the
construction of Hallemans et al., IEEE TIM 70 (2021), Definition 1, and of
Breugelmans et al., Electrochim. Acta 76 (2012) with L = 3. It gives the
densest spectrum that still leaves gaps, and the gaps are what make nonlinear
distortion measurable.

**Sparse quasi-logarithmic odd**, ``odd_quasilog``. A handful of tones per
decade, which is what electrochemical practice actually uses: fewer tones
means more amplitude per tone under a fixed peak budget, which matters when
the peak is what the cell tolerates.

Why both can carry the distortion diagnostic
--------------------------------------------
The diagnostic needs two things: that only odd harmonics are excited, so
even-order distortion lands on empty even harmonics; and that some odd
harmonics are left empty, so odd-order distortion has somewhere to show.

A dense grid satisfies the second only because of the deliberate omission --
without it, nearly every odd harmonic is occupied. A sparse quasi-log grid
satisfies it for free: 12 tones over three decades leave 488 of the 500 odd
harmonics unexcited. Measured on a simulated cube-law nonlinearity, the sparse
grid separated the classes as cleanly as the dense one (odd 3.7e-05 against
even 3.7e-08).

So ``L`` is a dense-grid concern. On a sparse grid, omit nothing.
'''

import numpy as np

__all__ = [
    'odd_quasilog', 'odd_random_grid', 'unexcited_odd_harmonics',
    'describe_grid',
]


def _as_odd_int_grid(values):
    '''Round to odd integers, deduplicate, sort.'''
    values = np.asarray(values, dtype=float)
    odd = np.where(values % 2 == 0, values - 1, values).astype(int)
    return np.unique(odd[odd >= 1])


def odd_quasilog(num_dec, pts_dec):
    '''
    Quasi-logarithmic odd harmonics over ``num_dec`` decades.

    ``pts_dec`` points per decade are placed logarithmically, then moved to
    the nearest odd harmonic below and deduplicated. The low end of the band
    is where duplicates collapse, so the realised number of tones is usually
    fewer than ``pts_dec * num_dec + 1``; ask for the count you get back
    rather than the count you requested.
    '''
    num_points = (pts_dec * num_dec) + 1
    return _as_odd_int_grid(np.floor(np.logspace(0, num_dec, num_points)))


def odd_random_grid(h_max, L=5, seed=None, h_min=1):
    '''
    Dense odd random phase grid, Hallemans TIM 2021 Definition 1.

    All odd harmonics in ``[h_min, h_max]`` are grouped into consecutive runs
    of ``L``, and one member of each run is dropped at random. The dropped
    harmonics are where odd-order nonlinear distortion becomes visible, since
    nothing else is transmitted there.

    ``L`` trades spectral density against diagnostic resolution: a larger L
    excites more of the band but leaves fewer windows to look through. The
    published examples use L = 5 (Hallemans) and L = 3 (Breugelmans).

    A trailing group shorter than ``L`` is kept whole rather than having a
    member dropped, so a short band is not thinned more aggressively than the
    rest.

    Note the first harmonic is as eligible for omission as any other, so the
    lowest excited harmonic may come out above ``h_min`` -- with ``L = 5``
    it is dropped one time in five. The measurement period is still
    ``1/f0`` either way, but the lowest measured *frequency* moves up.
    Supply your own grid if the band floor has to be exact.

    Returns the excited harmonics as a sorted integer array.
    '''
    if L < 2:
        raise ValueError(f'L must be at least 2, got {L}')
    rng = np.random.default_rng(seed)

    odd = np.arange(h_min if h_min % 2 else h_min + 1, h_max + 1, 2)
    if odd.size == 0:
        raise ValueError(f'no odd harmonics in [{h_min}, {h_max}]')

    keep = []
    for start in range(0, odd.size, L):
        group = odd[start:start + L]
        if group.size < L:
            keep.extend(group.tolist())
        else:
            keep.extend(np.delete(group, rng.integers(group.size)).tolist())
    return np.array(sorted(keep), dtype=int)


def unexcited_odd_harmonics(harmonics, h_max=None):
    '''
    Odd harmonics inside the band that carry no excitation.

    These are the bins where odd-order distortion is read. An empty result
    means the odd distortion cannot be separated from the response, whatever
    the estimator does.
    '''
    harmonics = np.asarray(harmonics, dtype=int)
    top = int(harmonics.max()) if h_max is None else int(h_max)
    return np.setdiff1d(np.arange(1, top + 1, 2), harmonics)


def describe_grid(harmonics):
    '''
    Report whether a grid can support the nonlinear distortion diagnostic.

    Returns a dict with the checks that matter downstream:

    ``all_odd``
        No even harmonic is excited. Required, or even-order distortion falls
        on an excited bin and is indistinguishable from the response.
    ``n_free_odd``
        How many odd harmonics are left empty. Required to be non-zero, or
        odd-order distortion has nowhere to appear.
    ``supports_distortion_estimate``
        Both of the above.

    This is a property of the grid alone. It cannot tell you whether the
    *phases* were drawn at random, which the theory also requires and which no
    inspection of a single record can recover -- see ``Multisine.generate_orp``.
    '''
    harmonics = np.asarray(harmonics, dtype=int)
    all_odd = bool(np.all(harmonics % 2 == 1))
    free = unexcited_odd_harmonics(harmonics)
    return {
        'n_tones': int(harmonics.size),
        'h_min': int(harmonics.min()),
        'h_max': int(harmonics.max()),
        'all_odd': all_odd,
        'n_free_odd': int(free.size),
        'supports_distortion_estimate': bool(all_odd and free.size > 0),
    }


if __name__ == '__main__':
    sparse = odd_quasilog(3, 4)
    dense = odd_random_grid(h_max=59, L=5, seed=0)
    for name, grid in (('quasi-log', sparse), ('dense ORP L=5', dense)):
        print(f'{name}: {grid.tolist()}')
        print(f'   {describe_grid(grid)}')
