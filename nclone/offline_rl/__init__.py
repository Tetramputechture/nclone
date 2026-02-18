"""Offline RL value learning from expert demonstrations.

This module implements IQL (Implicit Q-Learning) for learning V(s) and Q(s,a)
from N++ expert demonstrations, producing dense value functions that can be
used as PBRS potentials for FiGAR-PPO training.
"""
