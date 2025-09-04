# In this file I test that the gymnasium and the jax version of the environments are virtually the same
import gymnasium
import jax 
from jax import numpy as jnp
import sys

from tbu_gym.tbu_continous import TruckBackerEnv_C
from tbu_gym.tbu_discrete import TruckBackerEnv_D
from tbu_jax.tbu_continous import TBUax_c
from tbu_jax.tbu_discrete import TBUax_d

TEST_VERSION = "c"


# Fix the Randomness to deal with the different RNGs.
if TEST_VERSION == "c":
    # Making Gym Version 
    env_gym = TruckBackerEnv_C(seed=0)
    # Making Gymnax Version 
    env_jax = TBUax_c()
