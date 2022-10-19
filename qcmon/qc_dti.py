"""Code taken from qc_dti for use in datman.

#### Update the docs and test later.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend


def generate_directions_plot(bvec_file, bval_file, output_prefix):
    bvec = np.genfromtxt(bvec_file)
    bval = np.genfromtxt(bval_file)

    # HOTFIX: Account for small deviations in observed bvals
    bval[np.abs(bval) < 10] = 0
    idx_dirs = np.where(bval > 0)[0]

    xlim = abs_max(bvec[0, idx_dirs])
    ylim = abs_max(bvec[1, idx_dirs])
    zlim = abs_max(bvec[2, idx_dirs])

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(bvec[0, idx_dirs], bvec[1, idx_dirs], bvec[2, idx_dirs])
    ax.set_xlim3d([-xlim, xlim])
    ax.set_ylim3d([-ylim, ylim])
    ax.set_zlim3d([-zlim, zlim])
    plt.savefig(output_prefix + '_directions.png')


def abs_max(vector):
    """Finds the maximum absolute value of the input vector.
    """
    abs_max = np.abs(np.max(vector))
    abs_min = np.abs(np.min(vector))
    return np.max(abs_max, abs_min)
