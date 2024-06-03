import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scienceplots

from matplotlib.patches import Polygon

def debugger_is_active() -> bool:
    """Return if the debugger is currently active

    https://stackoverflow.com/a/67065084/7964333
    """
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

plt.style.use(['science','vibrant'])
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    # 'figure.figsize' : (8, 6),
    'figure.dpi' : 300,
    'hatch.linewidth' : 0.5,
})
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def scopus():
    df = pd.read_csv('scopus-search.csv', header=None, skiprows=9, names=['Year', 'Count'])

    plt.plot(df['Year'], df['Count'], 'o-', markersize=2)

    plt.xlabel('Year')
    plt.ylabel('Count')

    if debugger_is_active():
        plt.show()
    else:
        plt.savefig('pictures/scopus.pdf')

def milp_example():
    # Problem definition
    # min -x_2
    # s.t. x_1 <= x_2
    #      x_1 + x_2 <= 5
    #      x_2 >= 1
    #      x \in \Z^2

    # ax = plt.axes()
    # ax.add_patch(Polygon([[0., 0.], [2.5, 2.5], [5., 0.], [0., 0.]], facecolor='b', edgecolor='k'))
    # plt.fill([0., 2.5, 5., 0.], [0., 2.5, 0., 0.], colors[2], edgecolor=colors[1], label=r"$A\boldsymbol{x} \le \boldsymbol{b}$")
    # plt.fill([0., 2.5, 5., 0.], [0., 2.5, 0., 0.], colors[2], alpha=0.5, label=r"$A\boldsymbol{x} \le \boldsymbol{b}$")
    plt.fill([1., 2.5, 4., 1.], [1., 2.5, 1., 1.], facecolor='w', edgecolor=colors[2], linestyle='--', linewidth=0.5, label=r"$A\boldsymbol{x} \le \boldsymbol{b}$")
    x_feasible = np.arange(1, 5, 1)
    y_max = np.min([5 - x_feasible, x_feasible], 0)
    # plt.vlines(x_feasible, np.ones(len(x_feasible)), y_max, colors[1], 'solid', linewidth=2, capstyle='round', label=r"$\mathcal{X}$")
    plt.scatter(x_feasible, y_max, s=1, c=colors[1], label=r"$\mathcal{X}$")
    plt.scatter(x_feasible, np.ones(len(x_feasible)), s=1, c=colors[1])
    plt.xlim(0, 5.15)
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
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$', rotation=0)
    plt.gca().xaxis.set_label_coords(1.05, 0.04)
    plt.gca().yaxis.set_label_coords(0.00, 1.05)
    plt.legend()

    if debugger_is_active():
        plt.show()
    else:
        plt.savefig('pictures/milp_example_feasible_region.pdf')

def overfitting():
    np.random.seed(42)
    x_train = np.linspace(0.5, 4.5, 40)
    x_test = np.linspace(0, 5, 100)
    y = lambda x: (x-2)**2 + 1 + np.random.normal(0, 0.5, len(x))
    y_train = y(x_train)
    y_test = y(x_test)

    p1, res1, *_ = np.polyfit(x_train, y_train, 1, full=True)
    p3, res3, *_ = np.polyfit(x_train, y_train, 3, full=True)
    p15, res15, *_ = np.polyfit(x_train, y_train, 15, full=True)

    print(f'R_emp(f_1) = {res1[0]:.2f}')
    print(f'R_emp(f_2) = {res3[0]:.2f}')
    print(f'R_emp(f_3) = {res15[0]:.2f}')

    plt.plot(x_test, np.polyval(p1, x_test), label=r"$f_1$")
    plt.plot(x_test, np.polyval(p3, x_test), label=r"$f_2$")
    plt.plot(x_test, np.polyval(p15, x_test), label=r"$f_3$")

    plt.scatter(x_train, y_train, c='r', s=1, label=r'$\mathcal{D}$', zorder=10)

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim(0, 5)
    plt.ylim(0, 10)
    plt.legend()

    if debugger_is_active():
        plt.show()
    else:
        plt.savefig('pictures/overfitting.pdf')

def primal():
    np.random.seed(33)

    n = 100
    T = np.linspace(0, 10, n)
    Y = np.random.rand(n) * np.linspace(10, 1, n) + np.linspace(3, 1, n)

    for i in range(n):
        Y[i] = min(Y[:i+1])

    start = np.random.randint(0, 10)
    Y_area = Y
    Y_area[:start] = 5

    plt.step(T[start:], Y[start:], where='post', label=r"$c^T \hat{y}(t)$", zorder=100)
    # plt.fill_between(T, Y_area, 1, step='post', fc='white', hatch='///', label="$Primal\\,Integral$")
    plt.fill_between(T, Y_area, 1, step='post', fc='white', hatch='///', label=r"$\int c^T \hat{y}(t) dt$")
    plt.hlines(1, 0, 10, colors='black', linestyles='--', linewidth=0.5)
    plt.hlines(5, 0, 10, colors='black', linestyles='--', linewidth=0.5)
    plt.xlim(0, 11)
    plt.ylim(0, 6)

    plt.xlabel("$t$")
    plt.xticks([0, 10], ["$0$", "$T$"])
    # plt.yticks([0, 1, 3, 5], ["$0$", "$c^T y^*$", "$c^T y$", "$c^T \\overline{y}$"])
    plt.yticks([0, 1, 5], ["", "$c^T y^*$", "$c^T \\overline{y}$"])
    # plt.ylabel("$c^T y$", rotation=0)

    # formatting
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
    plt.gca().xaxis.set_label_coords(1.05, 0.04)
    # plt.gca().yaxis.set_label_coords(0.00, 1.05)
    plt.legend(loc=(0.4,0.5))

    if debugger_is_active():
        plt.show()
        print('debugger')
    else:
        plt.savefig('pictures/primal.pdf')

if __name__ == '__main__':
    try:
        incumbent_plot = sys.argv[1]
    except IndexError:
        print("Specify which figure to plot!")
        quit()

    if incumbent_plot in ['--all', '-a']:
        for plot in [scopus, milp_example, overfitting]:
            plt.figure()
            plot()
    else:
        eval(incumbent_plot + '()')
