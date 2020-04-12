"""
Interface of all ppo operations
"""
import argparse

import ppo_cartpole
import utils
from utils import ParamDict
from utils import demo_train

import time, os



# Build an argument parser
# use ArgumentParser class
parser = argparse.ArgumentParser()
# bound commandline parameters
#                       <----  Algorithm Parameters  ---->
parser.add_argument('--job_name', help='designate what to do', type=str, required=True)
parser.add_argument('--env_name', help='designate the Gym environment', type=str, required=False)
parser.add_argument('--hidden_dim', help='dimension of the hidden state in actor network', type=int, required=False)
parser.add_argument('--learning_rate', help='learning rate of policy update', type=float, required=False)
parser.add_argument('--batch_size', help='batch size for policy update', type=int, required=False)
parser.add_argument('--policy_epochs', help='number of epochs per policy update', type=int, required=False)
parser.add_argument('--entropy_coef', help='hyperparameter to vary the contribution of entropy loss',
                    type=float, required=False)
parser.add_argument('--critic_coef', help='Coefficient of critic loss when weighted against actor loss',
                    type=float, required=False)
parser.add_argument('--rollout_size', help='number of collected rollout steps per policy update',
                    type=int, required=False)
parser.add_argument('--num_updates', help='number of training policy iterations', type=int, required=False)
parser.add_argument('--discount', help='discount factor', type=float, required=False)
parser.add_argument('--plotting_iters', help='interval for logging graphs and policy rollouts',
                    type=int, required=False)
parser.add_argument('--n_seeds', help='number of training with different random seeds', type=int, required=False)
parser.add_argument('--parameters', help='path-filename of the file that stores initializing algorithm parameters',
                    type=int, required=False)
#                        <----  Result Saving Parameters  ---->
parser.add_argument('--show_plot', help='plot the learning curve if true', type=bool, required=False)
parser.add_argument('--save_plot', help='save the learning curve in ./save/img if true', type=bool, required=False)
parser.add_argument('--plot_path', help='learning curve save path string, linux style', type=str, required=False)
parser.add_argument('--plot_name', help='learning curve image name', type=str, required=False)
    # parser.add_argument('', help='', type=, required=)
# parse commandline arguments
args = parser.parse_args()
# record algorithm parameters
#                       <----  Algorithm Parameters  ---->
job_name = 'test' if args.job_name is None else args.job_name
env_name = 'CartPole-v1' if args.env_name is None else args.env_name
hidden_dim = 32 if args.hidden_dim is None else args.hidden_dim
learning_rate = 1e-3 if args.learning_rate is None else args.learning_rate
batch_size = 1024 if args.batch_size is None else args.batch_size
policy_epochs = 4 if args.policy_epochs is None else args.policy_epochs
entropy_coef = 0.001 if args.entropy_coef is None else args.entropy_coef
critic_coef = 0.5 if args.critic_coef is None else args.critic_coef
rollout_size = 2050 if args.rollout_size is None else args.rollout_size
num_updates = 3 if args.num_updates is None else args.num_updates
discount = 0.99 if args.discount is None else args.discount
plotting_iters = 2 if args.plotting_iters is None else args.plotting_iters
n_seeds = 3 if args.n_seeds is None else args.n_seeds
parameters = None
#                        <----  Result Saving Parameters  ---->
show_plot = False if args.show_plot is None else args.show_plot
save_plot = True if args.save_plot is None else args.save_plot
plot_path = os.path.join('.', 'save', 'img') if args.plot_path is None else args.plot_path
plot_name = 'training_res' + time.strftime("[%Y-%m-%d_%H:%M:%S]", time.localtime()) \
    if args.plot_name is None else args.plot_name



# Task Distribute
# test ppo_cartpole
if args.job_name == 'test':
    policy_params = ParamDict(
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        batch_size=batch_size,
        policy_epochs=policy_epochs,
        entropy_coef=entropy_coef,
        critic_coef=critic_coef
    )
    params = ParamDict(
        policy_params=policy_params,
        rollout_size=rollout_size,
        num_updates=num_updates,
        discount=discount,
        plotting_iters=plotting_iters,
        env_name=env_name,
    )
    test_ppo = ppo_cartpole.PPO(params)
    demo_train(test_ppo, n_seeds=n_seeds)
else:
    print("JOBError: can not recognize this job_name: {}".format(job_name))

# if __name__ == '__main__':
