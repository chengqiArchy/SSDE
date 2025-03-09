from typing import Dict, List
from flax.core import unfreeze, FrozenDict
import gym
import numpy as np
import jax.numpy as jnp


def evaluate(agent, env: gym.Env, num_episodes: int, with_task_embed=False, task_i=None) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            if with_task_embed:
                action = agent.sample_a_with_task_embed(observation[np.newaxis], temperature=0.0)
            else:
                if task_i is None:
                    action = agent.sample_actions(observation[np.newaxis], temperature=0.0)
                else:
                    action = agent.sample_actions(observation[np.newaxis], task_i, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'success' in info:
            if successes is None:
                successes = 0.0
            successes += info['success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats

def change_overlap_param_mask(agent, overlap_param, task_id, outside_task_id, multi_head, beta_dict=None):
    if task_id in overlap_param:
        a = agent.actor.params
        a = unfreeze(a)
        if multi_head:
            a['mean_layer']['kernel'] = multi_head[task_id]
        a['overlap_params_dict'] = overlap_param[task_id]
        a['beta'] = beta_dict[task_id]
        a = FrozenDict(a)
        agent.actor = agent.actor.replace(params=a)
    else:
        a = agent.actor.params
        a = unfreeze(a)
        if multi_head:
            a['mean_layer']['kernel'] = multi_head[task_id]
        a['overlap_params_dict'] = None # this is a placeholder..
        a = FrozenDict(a)
        agent.actor = agent.actor.replace(params=a)
        
    return agent

def evaluate_cl(agent, envs: List[gym.Env], num_episodes: int, current_task_id: int, overlap_param: dict, beta_dict: dict, multi_head=None, naive_sac=False, tadell=False) -> Dict[str, float]:
    stats = {}
    sum_return = 0.0
    sum_success = 0.0
    sum_success_final = 0.0
    list_log_keys = ['return']
    
    # dummy inputs
    # dummy_obs = jnp.ones((128, 12))

    for task_i, env in enumerate(envs):
        agent = change_overlap_param_mask(agent, overlap_param, task_i, outside_task_id=current_task_id, beta_dict=beta_dict, multi_head=multi_head)
            
        for k in list_log_keys:
            stats[f'{task_i}-{env.name}/{k}'] = []
        successes = None
        successes_final = None # if the task is success in the episode it will record as one

        if tadell:
            agent.select_actor(task_i)

        for _ in range(num_episodes):
            observation, done = env.reset(), False
            flag_success = 0
            while not done:
                
                if naive_sac:
                    action = agent.sample_actions(observation[np.newaxis], temperature=0)
                    action = np.asarray(action, dtype=np.float32).flatten()
                elif tadell:
                    action = agent.sample_actions(observation[np.newaxis], temperature=0, eval_mode=True)
                    action = np.asarray(action, dtype=np.float32).flatten()
                else:
                    action = agent.sample_actions(observation[np.newaxis], task_i, temperature=0)
                    action = np.asarray(action, dtype=np.float32).flatten()

                observation, _, done, info = env.step(action)
                if 'success' in info:
                    flag_success += info['success'] 

            for k in list_log_keys:
                stats[f'{task_i}-{env.name}/{k}'].append(info['episode'][k])

            if 'success' in info:
                if successes is None:
                    successes = 0.0
                successes += info['success']
            
            if flag_success > 0.5:
                if successes_final is None:
                    successes_final = 0.0
                successes_final += 1.0

        for k in list_log_keys:
            stats[f'{task_i}-{env.name}/{k}'] = np.mean(stats[f'{task_i}-{env.name}/{k}'])

        if successes is not None:
            stats[f'{task_i}-{env.name}/success'] = successes / num_episodes
            sum_success += stats[f'{task_i}-{env.name}/success']
        if successes_final is not None:
            stats[f'{task_i}-{env.name}/success_final'] = successes_final / num_episodes
            sum_success_final += stats[f'{task_i}-{env.name}/success_final']

        sum_return += stats[f'{task_i}-{env.name}/return']

        # stats[f'{task_i}-{env.name}/check_dummy_action'] = agent.sample_actions(dummy_obs, task_i, temperature=0).mean()

    stats['avg_return'] = sum_return / len(envs)
    stats['test/deterministic/average_success'] = sum_success / len(envs)
    stats['test/deterministic/average_success_final'] = sum_success_final / len(envs)

    agent = change_overlap_param_mask(agent, overlap_param,current_task_id, outside_task_id=current_task_id, beta_dict=beta_dict, multi_head=multi_head)

    return stats
