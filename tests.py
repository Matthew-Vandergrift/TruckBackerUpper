# In this file I test that the gymnasium and the jax version of the environments are virtually the same
import gymnasium
import jax 
from jax import numpy as jnp
import numpy as np
import sys

from tbu_gym.tbu_continous import TruckBackerEnv_C
from tbu_gym.tbu_discrete import TruckBackerEnv_D
from tbu_jax.tbu_continous import TBUax_c, EnvState
from tbu_jax.tbu_discrete import TBUax_d, EnvState

TEST_VERSION = "d"

def fix_state_gym(env):
    env.truck.x = 100
    env.truck.y = 0
    env.truck.theta_t = 0
    env.truck.theta_c = 0
    env.step_counter = 0
    return env

def fix_state_jax():
    env_state = EnvState(x=100,y=0,theta_t=0,theta_c=0,time=0,)
    return env_state


# Fix the Randomness to deal with the different RNGs.
if TEST_VERSION == "c":
    # Making Gym Version 
    env_gym = TruckBackerEnv_C(seed=0)
    # Making Gymnax Version 
    env_jax = TBUax_c()
    env_params = env_jax.default_params
elif TEST_VERSION == "d":
    # Making Gym Version 
    env_gym = TruckBackerEnv_D(seed=0)
    # Making Gymnax Version 
    env_jax = TBUax_d()
    env_params = env_jax.default_params

rng = jax.random.PRNGKey(0)
np.random.seed(0)

# Fix Truck Starting Locations
env_gym = fix_state_gym(env_gym)
env_state = fix_state_jax()

s = 0
while s <= 10_000:
    s += 1
    # Generate One Action for Both Environments
    if TEST_VERSION == "c":
        rng, _rng = jax.random.split(rng)
        action = jax.random.uniform(_rng, minval=-1.0, maxval=1.0)
    else:
        action = np.random.randint(0, 5+1)
    # Taking Action in Both Environments 
    rng, rng_step = jax.random.split(rng)
    obs_jax, env_state, reward_jax, done, info = env_jax.step(rng_step, env_state, action, env_params) # In JAX 
    obs_gym, reward_gym, terminated, truncated, info = env_gym.step(jnp.array([action])) # IN Gym     

    # Catching Internal Stochastic Transitions using the fact that I can look into the gym one 
    is_stochastic = (env_gym.had_recent_stochastic == True)
    if is_stochastic:
        env_gym = fix_state_gym(env_gym)
        env_state = fix_state_jax()
    # Resetting Environments one they are done to avoid different stochastic inits
    elif done or (terminated or truncated):
        env_gym = fix_state_gym(env_gym)
        env_state = fix_state_jax()
    else:
        # Checking for equality between outputs 
        obs_eq = np.allclose(obs_jax.squeeze(),obs_gym.squeeze(), rtol=0.01) 
        obs_reward = np.all(reward_gym == reward_jax)
        obs_term = np.all((terminated or truncated) == done)
        if not (obs_eq and obs_reward and obs_term):
            print("Failure on Step %s" %s)
            print("Terminal State :", (done or (terminated or truncated)))
            print("Failing Action :", action)
            if not obs_eq:
                print("Observation Failure")
                print(obs_gym, obs_jax)
            if not obs_reward:
                print("Reward Failure")
                print(reward_gym, reward_jax)
            if not obs_term:
                print("Termination Failure")
                print(terminated, truncated, done)
            sys.exit()
        else:
            print("made it through check :", s)
    # Resetting Environments one they are done to avoid different stochastic inits
    if done or (terminated or truncated):
        env_gym = fix_state_gym(env_gym)
        env_state = fix_state_jax()

print("Made it past the tests!")