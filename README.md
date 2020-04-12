## Implementation Plan

**Working on GitHub + Linux Server**

- [x] set a repo and organize folders/files

- [x] post the timeline on GitHub

- [x] set the remote machine and develop on local IDE

- [x] solve image saving problem and get familiar with the development tools

  - [x] fix current plot function & save
  - [x] view server images

-------

**PPO**

- **save image/video** to files
  - [ ] generate & save video
  - [ ] use tensorboard for monitoring loss
- design API & **wrap command-line** package (run on Linux server)

  - [x] build PPO interface
  - [ ] add expert until functions
    - [x] sample trajectories
    - [ ] ⋯
- **plots**(tensorboard) & save: 
  - reward 
  - entropy 
  - policy loss
  - value loss
- solve issues when running `CartPole-v1 `
  - change "success rate"
  - reward never goes up
- test: run on cartpole, debug, reformate code, etc.

<u>***&rArr; [aim, by Sunday]***</u>

- add code to run on **continuous envs**

  - understand how to modify algorithms
  - search on distribution class (e.g. categorical, gaussian, …)
  - design the api to share
  - implement
  - try on new envs:
    - 2d walker
    - humanoid

- code reformat/debug

- log critical points of implementation

  <u>***&rArr; [aim, by Monday]***</u>

----

**GAIL**

- **Implement GAIL** on simple Gym environments

  - ensure Gym environment(2d-Walker / cartpole / humanoid) can work and test on saving image/video
  - build trajectory storage/sample classes

  - build Generator & Discriminator classes

  - define training loop

    - controlling logic (sampling, updating networks)
    - compute parameters needed
    - compute loss function and update
    - check the training loop & make up utils functions needed

  - test: run on cartpole, debug, reformate code, etc.

  - implement logger utils, plot, save.

- Run on **more environments**

  - test on both discrete and continuous envs: 2d-Walker / cart-pole(humanoid)
  - modify code

- **Run Experiment & Plot**

  - read paper again and design the experiments(plots)
  - check the parameters and modify codes

  - run 2 critical experiment:

    - discrete: cartpole, Acrobot, **Mountain Car**
    - continuous: HalfCheetah, Hopper, Walker, Ant, **Humanoid**

  - save plot

  - (optional) run in jupyter and archive



## How to Use

**Install**

* ⋯
* `conda activate gail`

**Use PPO interface**

* `cd ppo`
* `python ppo.py --job_name test --num_updates [# training epoch] --plot_name [your training curv name]`



## Tech Log

**PyCharm**

* build ssh connection with remote server
* use SFTP for file sync
  * PyCharm: Tools &rarr; Deployment &rarr; Configuration
* Use remote interpreter
  * PyCharm: File &rarr; Settings(ctl + alt + s) &rarr; Project:[your proj name] &rarr; Project Interpreter &rarr; "settings icon" &rarr; Add... &rarr; SSH Interpreter
  * Linux server: `conda [your project env]` `which python`

**Other ways of viewing server files**

* use `http.server` service
  * in server folder: `python -m http.server`
  * in windows terminal: `ssh -N -L localhost:8000:localhost:8000 ziang@lim001.usc.edu`
  * open `http://127.0.0.1:8000/` in local browser



## Dev Log

`[4.10]`

* make impl plan
* start new repo
* connect server
* fix `FlatObsWrapperCartPole` fit `CartPole-v1` env
* stuck on image can not save issue

`[4.11]`

* learn `matplotlib` & `seaborn`, rewrite `def plot_grid_std_learning_curves` function to fit `CartPole-v1` env
* use PyCharm SFTP to replace Git for file sync
* wrapped a PPO interface for future use
* add more APIs (for ppo-expert in GAIL)
  * render trajectories & save
* find training issue: `CartPole-v1` reward does not go up:
  * <img src="./demo_trianing_res(2020-04-12_01:03:46).png
  * " style="zoom: 67%;" />

`[near future plans]`

* **finish designing API & build wrapping version**
* **plot**
  * modify/add plot functions
  * plot different losses
  * save video
  * use tensorboard
* fix/reformat code
  * fix current issue of reward never goes up
  * modify code to fit continuous environments
  * use GPU
* **fit continuous envs**
* start a code base
* ⋯