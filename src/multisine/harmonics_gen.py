'''
A module to genreate harmonics series
'''

import numpy as np

def odd_quasilog(num_dec, pts_dec):

    # Generate purly logarithmic distribution
    num_points = (pts_dec * num_dec) + 1
    log_dist = np.floor(np.logspace(0, num_dec, num_points))
    print(f'Logarithmic distribution:{log_dist}')

    # Remove duplicates
    log_dist = np.unique(log_dist)
    print(f'Logarithmic distribution unique :{log_dist}')

    # Make the distribution odd
    odd_quasilog_dist = list()
    for i, h in enumerate(log_dist):
        if h%2 != 0:
            odd_quasilog_dist.extend([h])
            print(f'{h} kept')
        elif h%2 == 0:
            odd_quasilog_dist.extend([h-1])
            print(f'{h} converted to {h-1}')
    odd_quasilog_dist = np.array(odd_quasilog_dist)
    # Remove duplicates
    odd_quasilog_dist = np.unique(odd_quasilog_dist)
    print(f'Quasi-log odd distribution: {odd_quasilog_dist}')
    return odd_quasilog_dist


if __name__ == '__main__':
    odd_quasilog(5, 5)