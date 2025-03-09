from jax.numpy import ndarray
import jax.numpy as jnp
from jaxrl.agents.sac.sac_learner import CoTASPLearner
# from sac_learner import CoTASPLearner
from typing import Any
import numpy as np
from jaxrl.dict_learning.task_dict import OnlineDictLearnerV2
from flax.core import freeze, unfreeze, FrozenDict

class RandomGenerateD(OnlineDictLearnerV2):
    pass


class MaskCombinationLearner(CoTASPLearner):
    def __init__(self, seed: int, observations: jnp.ndarray, actions: jnp.ndarray, task_num: int, load_policy_dir: str | None = None, load_dict_dir: str | None = None, update_dict=True, update_coef=True, dict_configs_fixed: dict = ..., dict_configs_random: dict = ..., pi_opt_configs: dict = ..., q_opt_configs: dict = ..., t_opt_configs: dict = ..., actor_configs: dict = ..., critic_configs: dict = ..., tau: float = 0.005, discount: float = 0.99, target_update_period: int = 1, target_entropy: float | None = None, init_temperature: float = 1):
        super().__init__(seed, observations, actions, task_num, load_policy_dir, load_dict_dir, update_dict, update_coef, dict_configs_fixed, pi_opt_configs, q_opt_configs, t_opt_configs, actor_configs, critic_configs, tau, discount, target_update_period, target_entropy, init_temperature)
        # fixed dictionary this has been integrated by CoTASP without any dictionary learning
        # random dictionary
        self.dict4layers_random = {}
        self.actor_configs = actor_configs
        self.dict_configs_random = dict_configs_random
        self.seed = seed
        
        
    
    def start_task(self, task_id: int, description: str):
        task_e = self.task_encoder.encode(description)[np.newaxis, :]
        self.task_embeddings.append(task_e)

        # set initial alpha for each layer of MPN
        actor_params = unfreeze(self.actor.params)
        for k in self.actor.params.keys():
            if k.startswith('embeds'):
                alpha_l = self.dict4layers[k].get_alpha(task_e)
                alpha_l = jnp.asarray(alpha_l.flatten())
                # Replace the i-th row
                actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(alpha_l)

        linear_range = np.linspace(0.01, 0.0001, 10)
        # linear_range = [0.01, 0.01, 0.01, 0.01, 0.00005, 0.001, 0.001, 0.001, 0.001, 0.001]
        actor_configs = self.actor_configs
        dict_configs = self.dict_configs_random
        for id_layer, hidn in enumerate(actor_configs['hidden_dims']):
            dict_configs['alpha'] = linear_range[task_id]
            # if task_id < 5:
            #     dict_configs['alpha'] = 0.01
                
            dict_learner = OnlineDictLearnerV2(
                384,
                hidn,
                task_id * 10000 + id_layer + 10000 + self.seed * 512,
                None,
                **dict_configs)
            self.dict4layers_random[f'random_embeds_bb_{id_layer}'] = dict_learner
        
        for k in self.actor.params.keys():
            if k.startswith('random'):
                alpha_l = self.dict4layers_random[k].get_alpha(task_e)
                alpha_l = jnp.asarray(alpha_l.flatten())
                # Replace the i-th row
                actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(alpha_l)
        self.actor = self.actor.update_params(freeze(actor_params))

    
    
    
