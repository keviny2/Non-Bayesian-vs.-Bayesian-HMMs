from simulate import SimulateData
import seaborn as sns
import matplotlib.pyplot as plt
import os

def KDEplot():
    data = SimulateData()
    obs, _ = data.simulate_continuous(num_obs = 1200)
    sns.set_style('whitegrid')
    sns.kdeplot(obs)
    sns.distplot(obs, hist = True)
    plt.show()
    # fname = os.path.join("..", "plots", "KDE")
    # plt.savefig(fname)
    # print("\nFigure saved as '%s'" % fname)


