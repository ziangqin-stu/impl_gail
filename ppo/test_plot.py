import glob
import io
import base64
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=2, suppress=True)

import matplotlib
import matplotlib.pyplot as plt

import os, time

from IPython.display import HTML
from IPython import display as ipythondisplay

f1 = [[18.581818181818182,
       20.896907216494846,
       23.77906976744186,
       20.663265306122447,
       22.0,
       24.228915662650603,
       19.99009900990099,
       21.73404255319149,
       20.43,
       22.96590909090909,
       21.69148936170213,
       19.34285714285714,
       18.925925925925927,
       22.197802197802197,
       18.554545454545455,
       22.47252747252747,
       20.636363636363637,
       20.505050505050505,
       19.825242718446603,
       19.37142857142857],
      [19.601941747572816,
       20.4,
       20.646464646464647,
       19.20754716981132,
       18.743119266055047,
       19.05607476635514,
       18.944444444444443,
       19.88235294117647,
       18.65137614678899,
       19.409523809523808,
       19.567307692307693,
       18.381818181818183,
       18.97222222222222,
       18.088495575221238,
       18.169642857142858,
       17.95614035087719,
       18.59090909090909,
       18.44144144144144,
       18.5,
       17.218487394957982],
      [19.0,
       15.4,
       20.0,
       17.20754716981132,
       18.743119266055047,
       19.0,
       18.0,
       19.0,
       15.65137614678899,
       19.0,
       19.0,
       18.0,
       18.0,
       16.088495575221238,
       18.0,
       17.95614035087719,
       19.0,
       18.0,
       14.5,
       18.218487394957982]]
f2 = [[x * -1.5 for x in f1[0]], [x * -1.5 for x in f1[1]], [x * -1.5 for x in f1[2]]]

import time
def plot_std_learning_curves(rewards, success_rates, num_it,
                             no_show=False, save=False, save_path='.\save\\', plot_name=None):
    r, sr = np.asarray(rewards), np.asarray(success_rates)
    df = pd.DataFrame(r).melt()
    sns.lineplot(x="variable", y="value", data=df, label='reward/eps')
    df = pd.DataFrame(sr).melt()
    sns.lineplot(x="variable", y="value", data=df, label='success rate')
    plt.xlabel("Training iterations")
    plt.ylabel("")
    plt.xlim([0, num_it])
    plt.legend()
    plt.grid('on')

    if not no_show:
        plt.show()

    if save:
        if plot_name is None:
            plot_name = time.strftime("%Y-%m-%d[%H:%M:%S]", time.localtime())
        plt.savefig('img.png')
    plt.close("all")


# plot_std_learning_curves(f1, f2, 20, save=True)
t = np.linspace(-np.pi, np.pi, 201)
sin = [np.sin(t).tolist(), [y + 0.5 for y in np.sin(t)], [y - 0.5 for y in np.sin(t)]]
exp = [np.exp(t).tolist(), [y + 0.5 for y in np.exp(t)], [y - 0.5 for y in np.exp(t)]]

def plot_std_learning_curves_debug(rewards, success_rates, num_it, no_show=False, plot_save=True, plot_name=None):
    # data prepare
    r, sr = np.asarray(rewards), np.asarray(success_rates)
    df1 = pd.DataFrame(r).melt()
    df2 = pd.DataFrame(sr).melt()

    # canvas/plot
    fig, ax1 = plt.subplots()
    color1 = 'blue'
    ln1 = sns.lineplot(x="variable", y="value", data=df1, label='reward/eps',
                 color=color1,legend=False)
    ax1.set_ylabel('reward/eps', color=color1)
    ax1.set_xlabel('Training iterations')
    # second subplot
    ax2 = ax1.twinx()
    color2 = 'orange'
    ln2 = sns.lineplot(x="variable", y="value", data=df2, label='success rate',
                 color='orange', legend=False)
    ax2.set_ylabel('success rate', color=color2)
    # merge legend
    # plt.legend()
    # change tick color
    ax2.spines['left'].set_color(color1)
    ax2.spines['right'].set_color(color2)
    if plot_save:
        plot_name = plot_name if plot_name is not None else time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        plt.savefig(os.path.join('.', 'save', 'img', plot_name + '.png'))
    if not no_show:
        plt.show()

plot_std_learning_curves_debug(sin, exp, len(t), plot_save=False)