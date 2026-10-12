'''
A module that provides functions for the optimization of phases to reduce
the crest factor of a multisine waveform.
'''

import numpy as np
from scipy.optimize import minimize


def compute_multisine(phases, freq, ampli, sampling_frequency, num_points):
    time = np.arange(num_points) / sampling_frequency
    return np.sum(
        ampli[:, None] * np.sin(
            2 * np.pi * freq[:, None] * time[None, :] + phases[:, None]
        ),
        axis=0
    )


def lp_norm_objective(phases, freq, ampli, sampling_frequency, num_points, p=8):
    '''
    Smooth surrogate for the crest factor's peak term.

    RMS of an orthogonal multisine is invariant to phase (Parseval), so a
    least-squares fit of the waveform itself cannot change the crest
    factor - only the peak matters, and max(|x|) isn't differentiable.
    The Lp norm approaches max(|x|) as p grows while staying smooth, so
    it works as a gradient-friendly objective.
    '''
    waveform = compute_multisine(phases, freq, ampli, sampling_frequency, num_points)
    return np.sum(np.abs(waveform) ** p) ** (1.0 / p)


def optimize_phases_crest(
        freq: np.ndarray,
        ampli: np.ndarray,
        phase0: np.ndarray,
        sampling_frequency: float,
        num_points: int,
        p: float = 8,
        method: str = 'L-BFGS-B',
        **minimize_kwargs,
    ):
    '''
    Find phases that minimize the crest factor of a multisine, keeping
    frequencies and amplitudes fixed. Minimizes the Lp-norm surrogate
    with a gradient-based solver (default L-BFGS-B).
    '''
    result = minimize(
        lp_norm_objective,
        x0=phase0,
        args=(freq, ampli, sampling_frequency, num_points, p),
        method=method,
        **minimize_kwargs,
    )
    return result


def optimize_phases_crest_continuation(
        freq: np.ndarray,
        ampli: np.ndarray,
        phase0: np.ndarray,
        sampling_frequency: float,
        num_points: int,
        p_schedule=(2, 4, 8, 16, 32),
        method: str = 'L-BFGS-B',
        **minimize_kwargs,
    ):
    '''
    Same as optimize_phases_crest, but ramps p up gradually, using each
    solution as the warm start for the next. Helps avoid getting stuck
    when starting directly at a high (sharply-peaked, harder to
    optimize) p.
    '''
    phases = np.asarray(phase0, dtype=float)
    result = None
    for p in p_schedule:
        result = minimize(
            lp_norm_objective,
            x0=phases,
            args=(freq, ampli, sampling_frequency, num_points, p),
            method=method,
            **minimize_kwargs,
        )
        phases = result.x
    return result


def crest_factor(signal):
    return np.max(np.abs(signal)) / np.sqrt(np.mean(signal ** 2))


if __name__ == '__main__':
    freq = np.array([1,3,5,9,15,25,39,63,99,157,251,397,], dtype=float)
    ampli = np.ones_like(freq)
    sampling_frequency = 1000.0
    num_points = 100  # one period at the lowest frequency (1 Hz)

    rng = np.random.default_rng(0)
    phase0 = rng.uniform(0, 2 * np.pi, size=freq.size)

    waveform0 = compute_multisine(phase0, freq, ampli, sampling_frequency, num_points)
    print(f'Starting crest factor: {crest_factor(waveform0):.4f}')

    result = optimize_phases_crest_continuation(
        freq, ampli, phase0, sampling_frequency, num_points,
    )
    waveform1 = compute_multisine(result.x, freq, ampli, sampling_frequency, num_points)
    print(f'Optimized crest factor: {crest_factor(waveform1):.4f}')
    print(f'Optimized phases (rad): {result.x}')
