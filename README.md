## Implementation Plan

**Working on GitHub**

- [ ] set a repo and organize folders/files

- [ ] post the timeline on GitHub

- [ ] set the remote env and develop on local PyCharm

- [ ] solve the saving image/video problem and get familiar with the developing style 

  <u>***&rArr; [by Friday]***</u>

-------

**PPO**

- **save image/video** to files

- design API & **wrap command-line** package (run on Linux server)

- **plots**(tensorboard) & save: 

- - reward 
  - entropy 
  - policy loss
  - value loss

- test: run on cartpole, debug, reformate code, etc.

- add code to run on **continuous envs**

- - understand how to modify algorithms

  - search on distribution class (e.g. categorical, gaussian, …)

  - design the api to share

  - implement

  - try on new envs:

  - - 2d walker
    - humanoid

- code reformat/debug

- log critical points of implementation

  <u>***&rArr; [aim, by Monday]***</u>

----

**GAIL**

- **Implement GAIL** on simple Gym environments

- - ensure Gym environment(2d-Walker / cartpole / humanoid) can work and test on saving image/video

  - build trajectory storage/sample classes

  - build Generator & Discriminator classes

  - define training loop

  - - controlling logic (sampling, updating networks)
    - compute parameters needed
    - compute loss function and update
    - check the training loop & make up utils functions needed

  - test: run on cartpole, debug, reformate code, etc.

  - implement logger utils, plot, save.

- Run on **more environments**

- - test on both discrete and continuous envs: 2d-Walker / cart-pole(humanoid)
  - modify code

- **Run Experiment & Plot**

- - read paper again and design the experiments(plots)

  - check the parameters and modify codes

  - run 2 critical experiment:

  - - discrete: cartpole, Acrobot, **Mountain Car**
    - continuous: HalfCheetah, Hopper, Walker, Ant, **Humanoid**

  - save plot

  - (optional) run in jupyter and archive