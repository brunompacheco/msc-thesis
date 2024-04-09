import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scienceplots

from matplotlib.patches import Polygon

plt.style.use(['science','vibrant'])
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    # 'figure.figsize' : (8, 6),
    'figure.dpi' : 300,
})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


if __name__ == '__main__':
    df = pd.read_csv('scopus-search.csv', header=None, skiprows=9, names=['Year', 'Count'])

    plt.plot(df['Year'], df['Count'], 'o-', markersize=2)

    plt.xlabel('Year')
    plt.ylabel('Count')

    # plt.show()
    plt.savefig('pictures/scopus.pdf')
    plt.figure()

    # Problem definition
    # min -y
    # s.t. z <= y
    #      z + y <= 5
    #      y >= 0
    #      z \in \Z

    # ax = plt.axes()
    # ax.add_patch(Polygon([[0., 0.], [2.5, 2.5], [5., 0.], [0., 0.]], facecolor='b', edgecolor='k'))
    # plt.fill([0., 2.5, 5., 0.], [0., 2.5, 0., 0.], colors[2], edgecolor=colors[1], label=r"$A\boldsymbol{x} \le \boldsymbol{b}$")
    # plt.fill([0., 2.5, 5., 0.], [0., 2.5, 0., 0.], colors[2], alpha=0.5, label=r"$A\boldsymbol{x} \le \boldsymbol{b}$")
    plt.fill([0., 2.5, 5., 0.], [0., 2.5, 0., 0.], facecolor='w', edgecolor=colors[2], linestyle='--', linewidth=0.5, label=r"$A\boldsymbol{x} \le \boldsymbol{b}$")
    x_feasible = np.arange(0, 5, 1)
    y_max = np.min([5 - x_feasible, x_feasible], 0)
    plt.vlines(x_feasible, np.zeros(len(x_feasible)), y_max, colors[1], 'solid', capstyle='round', label=r"$\mathcal{X}$")
    plt.ylim(0, 4.5)
    plt.xticks(np.arange(0, 6, 1))

    plt.gca().spines[["left", "bottom"]].set_position(("data", 0))
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.gca().tick_params(
        axis='both',
        direction='inout',
        which='major',
        top=False,
        right=False,
        bottom=True,
        left=True,
    )
    plt.gca().tick_params(
        axis='both',
        which='minor',
        top=False,
        right=False,
        bottom=False,
        left=False,
    )
    xlim = plt.xlim()
    plt.gca().spines["bottom"].set_bounds(low=0, high=xlim[1])
    ylim = plt.ylim()
    plt.gca().spines["left"].set_bounds(low=0, high=ylim[1])
    plt.plot(1, 0, ">k", markersize=2, transform=plt.gca().get_yaxis_transform(), clip_on=False)
    plt.plot(0, 1, "^k", markersize=2, transform=plt.gca().get_xaxis_transform(), clip_on=False)
    plt.xlabel('$z$')
    plt.ylabel('$y$')
    plt.legend()

    # plt.show()
    plt.savefig('pictures/milp_example_feasible_region.pdf')
