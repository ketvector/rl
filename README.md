This repository hosts my (ongoing) implementations of RL algorithms. 
I am using 
- [gymnasium](https://gymnasium.farama.org/) for the environments. 
- pytorch for implementing the algorithm.

Implementations:

- I have implemented Policy Gradient with Baseline in [vpg.py](src/vpg/vpg.py). The weights for trained policy are stored in [saved-models](src/vpg/saved-models/vpg.pth). You can run [test.py](src/vpg/test.py) to run the agent with the saved policy. 