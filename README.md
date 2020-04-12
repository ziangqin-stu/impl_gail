## Implementation Plan

**Working on GitHub + Linux Server**

- [x] set a repo and organize folders/files

- [x] post the timeline on GitHub

- [x] set the remote env and develop on local PyCharm

  * failed to mount server folder on windows &rarr; using git to sync files
  * git failed to sync between different file systems &rarr; using PyCharm SFTP to sync code

- [ ] solve the saving image/video problem and get familiar with the developing style 

  - [ ] fix current plot function (learn seaborn)
    * `seanborn`, `matplotlib`
    * multi y-axis, merge legends, modify plot attributes 
  - [x] write saving util functions (issue)
    * cannot save using remote interpreter
    * success saved when running on server 
  - [ ] save video
  - [ ] use tensorboard

  <u>***&rArr; [by Friday]***</u>

-------

**PPO**

- **save image/video** to files

- solve issues when running `CartPole-v1 `

- design API & **wrap command-line** package (run on Linux server)

  - build PPO interface
  - reformat modify interface file

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