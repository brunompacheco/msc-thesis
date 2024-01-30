import matplotlib.pyplot as plt
import pandas as pd

import scienceplots


plt.style.use(['science','ieee','vibrant'])


if __name__ == '__main__':
    df = pd.read_csv('scopus-search.csv', header=None, skiprows=9, names=['Year', 'Count'])

    plt.plot(df['Year'], df['Count'], 'o-', markersize=2)

    plt.xlabel('Year')
    plt.ylabel('Count')

    # plt.show()
    plt.savefig('pictures/scopus.pdf', dpi=300)
