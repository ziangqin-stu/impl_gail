import glob
import io
import base64
import numpy as np
import pandas as pd
import seaborn as sns
np.set_printoptions(precision=2, suppress=True)

import matplotlib
import matplotlib.pyplot as plt

from IPython.display import HTML
from IPython import display as ipythondisplay

import time, os


"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""


def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_model_params(model):
    return sum(p.numel() for p in model.parameters())


def plot_grid_std_learning_curves(d, num_it):
    for i, key in enumerate(d):
        ax = plt.subplot(2, 2, i+1)
        rewards, success_rates = d[key]
        plot_std_learning_curves(rewards, success_rates, num_it, no_show=True)
        ax.set_title(key)
    plt.show()


def plot_std_learning_curves(rewards, success_rates, num_it, no_show=False, save=True, plot_name=None):
    print('Plotting...')
    # data prepare
    r, sr = np.asarray(rewards), np.asarray(success_rates)
    df1 = pd.DataFrame(r).melt()
    df2 = pd.DataFrame(sr).melt()

    # canvas/plot
    fig, ax1 = plt.subplots()
    color1 = 'blue'
    ln1 = sns.lineplot(x="variable", y="value", data=df1, label='reward/eps',
                       color=color1, legend=False)
    ax1.set_ylabel('reward/eps', color=color1)
    ax1.set_xlabel('Training iterations')

    ax2 = ax1.twinx()
    color2 = 'orange'
    ln2 = sns.lineplot(x="variable", y="value", data=df2, label='success rate',
                       color='orange', legend=False)
    ax2.set_ylabel('success rate', color=color2)

    # modify legend & axis tcket color
    if save:
        print("Saving training curve...")
        plot_name = plot_name if plot_name is not None else time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        plt.savefig(os.path.join('.', 'save', 'img', plot_name + '.png'))
        print("Training curve saved: {}".format(os.path.join('.', 'save', 'img', plot_name + '.png')))
    if not no_show:
        plt.show()


def plot_learning_curve(rewards, success_rate, num_it, plot_std=False, plot_name=None):
    if plot_std:
        # plots shaded regions if list of reward timeseries is given
        plot_std_learning_curves(rewards, success_rate, num_it, plot_name=plot_name)
    else:
        plt.plot(rewards, label='reward/eps')
        if success_rate:
            plt.plot(success_rate, label='success rate')
            plt.legend()
        else:
            plt.ylabel('return / eps')
        plt.ylim([0, 1])
        plt.xlim([0, num_it - 1])
        plt.xlabel('train iter')
        plt.grid('on')
        plt.show()


def demo_train(ppo, n_seeds=3,
               plot_name='demo_trianing_res'+time.strftime("[%Y-%m-%d_%H:%M:%S]", time.localtime())):
    # trian PPO policy
    rewards, success_rates = [], []
    for i in range(n_seeds):
        print("Start training run {}!".format(i))
        r, sr = ppo.train(i)
        rewards.append(r)
        success_rates.append(sr)
    print('All training runs completed!')
    # plot training curv
    plot_learning_curve(rewards, success_rates, ppo.params.num_updates, plot_std=True, plot_name=plot_name)
    print("Average Reward: {}".format(np.mean(rewards, axis=0)[-1]))
    print("Average Success Rates: {}".format(np.mean(success_rates, axis=0)[-1]))

# Wrapper around dictionary that enables attribute access instead of the bracket syntax
# i.e. you can replace d['item'] with d.item
class ParamDict(dict):
    __setattr__ = dict.__setitem__
    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)
    def __getstate__(self): return self
    def __setstate__(self, d): self = d