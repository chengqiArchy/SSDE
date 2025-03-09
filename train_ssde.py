'''
CONTINUAL TASK ALLOCATION IN META-POLICY NETWORK VIA SPARSE PROMPTING
'''

import itertools
import random
import time
import jax

import numpy as np
import wandb
import yaml
from absl import app, flags
from ml_collections import config_flags, ConfigDict

from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate_cl
from jaxrl.utils import Logger
from jaxrl.agents.sac.sac_learner import CoTASPLearner
from jaxrl.agents.sac.sac_mask_combination import MaskCombinationLearner
from continual_world import TASK_SEQS, get_single_env
import jax.numpy as jnp
from jax.tree_util import tree_map
from flax.core import unfreeze, FrozenDict
from functools import partial
from jaxrl.agents.sac.sac_learner import MPNTrainState
import pickle
from jaxrl.networks.common import default_init

FLAGS = flags.FLAGS
# flags.DEFINE_string('env_name', 'cw1-stick-pull', 'Environment name.')
# flags.DEFINE_string('env_name', 'cw3-test', 'Environment name.')
flags.DEFINE_string('env_name', 'cw10', 'Environment name.')
flags.DEFINE_integer('seed', 220, 'Random seed.')
flags.DEFINE_string('base_algo', 'cotasp', 'base learning algorithm')

flags.DEFINE_string('env_type', 'random_init_all', 'The type of env is either deterministic or random_init_all')
flags.DEFINE_boolean('normalize_reward', True, 'Normalize rewards')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 200, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updating per # environment steps.')
flags.DEFINE_integer('buffer_size', int(1e6), 'Size of replay buffer')
flags.DEFINE_integer('max_step', int(1e6), 'Number of training steps for each task')
flags.DEFINE_integer('start_training', int(1e4), 'Number of training steps to start training.')
flags.DEFINE_integer('theta_step', int(990), 'Number of training steps for theta.')
flags.DEFINE_integer('alpha_step', int(10), 'Number of finetune steps for alpha.')

flags.DEFINE_boolean('rnd_explore', True, 'random policy distillation')
flags.DEFINE_integer('distill_steps', int(2e4), 'distillation steps')

flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_string('wandb_mode', 'online', 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "beta mechanism multi head", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', None, "the entity (team) of wandb's project")
flags.DEFINE_boolean('save_checkpoint', False, 'Save meta-policy network parameters')
flags.DEFINE_string('save_dir', '~/rl-archy/Documents/PyCode/CoTASP/logs', 'Logging dir.')


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>> multi-head >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
flags.DEFINE_bool('multi_head', False, 'whether to use multi-head in the actor')
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>> beta mechanism >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
flags.DEFINE_bool('use_adaptive_beta', False, 'whether to use adaptive beta')
flags.DEFINE_float('beta_lambda', 0.5, 'the beta lambda for the beta mechanism')
flags.DEFINE_float('default_beta', 1, 'the value of default beta')
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>> input sensitive >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
flags.DEFINE_bool('use_input_sensitive', False, 'whether to use input sensitive')
flags.DEFINE_integer('calculate_layer_sensitivity_interval', int(8e4), 'calculate the layer sensitivity every x steps')
flags.DEFINE_integer('evaluation_batch_size', int(1e3), "the batch size for evaluation")
flags.DEFINE_float('layer_neuron_threshold', 0.6, 'the threshold to reset the parameters')
flags.DEFINE_integer('stop_reset_after_steps', int(6.5e5), 'stop reset once the steps reach this value')

flags.DEFINE_bool('is_store_everything', False, 'store everything')
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Debug >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
flags.DEFINE_bool('use_quick_experiments', False, 'set the first four task quicker')
flags.DEFINE_integer('first_four_task_max_steps', int(5e5), 'set the first four task max steps')
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> reset log_std >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
flags.DEFINE_bool('reset_log_std', False, 'reset the log_std')


# YAML file path to cotasp's hyperparameter configuration
with open('configs/sac_cotasp.yaml', 'r') as file:
    yaml_dict = yaml.unsafe_load(file)
config_flags.DEFINE_config_dict(
    'config',
    ConfigDict(yaml_dict),
    'Training hyperparameter configuration.',
    lock_config=False
)

def main(_):
    # config tasks
    seq_tasks = TASK_SEQS[FLAGS.env_name]
    algo_kwargs = dict(FLAGS.config)
    algo = FLAGS.base_algo
    run_name = f"{FLAGS.env_name}__{algo}__{FLAGS.seed}__fixed_lambda:{algo_kwargs['dict_configs_fixed']['alpha']}__random_lambda:{algo_kwargs['dict_configs_random']['alpha']}__{int(time.time())}"

    if FLAGS.save_checkpoint:
        save_policy_dir = f"logs/saved_actors/{run_name}.json"
        save_dict_dir = f"logs/saved_dicts/{run_name}"
    else:
        save_policy_dir = None
        save_dict_dir = None

    wandb.init(
        project=FLAGS.wandb_project_name,
        entity=FLAGS.wandb_entity,
        sync_tensorboard=True,
        config=FLAGS,
        name=run_name,
        monitor_gym=False,
        save_code=False,
        mode=FLAGS.wandb_mode,
        dir=FLAGS.save_dir
    )
    wandb.config.update({"algo": algo})

    log = Logger(wandb.run.dir)

    # random numpy seeding
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # initialize SAC agent
    algo_kwargs['update_coef'] = False
    algo_kwargs['update_dict'] = False
    temp_env = get_single_env(
        TASK_SEQS[FLAGS.env_name][0]['task'], FLAGS.seed, 
        randomization=FLAGS.env_type)
    if algo == 'cotasp':
        # agent = CoTASPLearner(
        agent = MaskCombinationLearner(
            FLAGS.seed,
            temp_env.observation_space.sample()[np.newaxis],
            temp_env.action_space.sample()[np.newaxis], 
            # len(seq_tasks),
            20,
            **algo_kwargs)
        del temp_env
    else:
        raise NotImplementedError()
    
    '''
    continual learning loop
    '''
    eval_envs = []
    for idx, dict_task in enumerate(seq_tasks):
        eval_envs.append(get_single_env(dict_task['task'], FLAGS.seed, randomization=FLAGS.env_type))

    total_env_steps = 0
    overlap_params_dict = {}
    beta_dict = {}
    get_beta = lambda frozen_number, total_number: min((total_number - frozen_number) / frozen_number, 1)

    
    if FLAGS.multi_head:
        head_tamplate = unfreeze(agent.actor.params['mean_layer']['kernel'].copy())
        head_dict = {}
        for i in range(len(seq_tasks)):
            key = jax.random.PRNGKey(i)
            head_dict[i] = default_init()(key, head_tamplate.shape)
    
    log_std_params = None
    for task_idx, dict_task in enumerate(seq_tasks):
        
        '''
        Learning subroutine for the current task
        '''
        print(f'Learning on task {task_idx+1}: {dict_task["task"]} for {FLAGS.max_step} steps')
        # start the current task
        agent.start_task(task_idx, dict_task["hint"])
        if FLAGS.reset_log_std:
            if log_std_params is None:
                log_std_params = agent.actor.params['log_std_layer']
            else:
                a = unfreeze(agent.actor.params)
                a['log_std_layer'] = log_std_params
                agent.actor = agent.actor.replace(params=FrozenDict(a))
            
            

        # >>>>>>>>>>>>>>>>>>>> log parameters situations >>>>>>>>>>>>>>>
        if FLAGS.multi_head:
            a = unfreeze(agent.param_masks)
            a[('mean_layer', 'kernel')] = jnp.ones_like(agent.actor.params['mean_layer']['kernel'])
            agent.param_masks = FrozenDict(a)

        a, b = agent.actor(agent.dummy_o, jnp.array([task_idx]))
        current_grad_masks = agent.get_grad_masks(
            {'params': agent.actor.params}, b['masks']
        )
        overlap_params = tree_map(lambda x, y: (1 - x) * (1 - y), current_grad_masks, unfreeze(agent.param_masks))
        overlap_params_dict[task_idx] = overlap_params
        agent.actor = agent.actor.replace(overlap_params_dict=overlap_params_dict)

            
        # calculate the overlap parameters, and log it.
        layer_name_list = ['backbones_0', 'backbones_1', 'backbones_2', 'backbones_3', 'mean_layer']
        overlap_params_number = {}
        forward_params_number = {}
        total_overlap_params = 0
        total_forward_params = 0
        current_beta = {}
        for layer_name in layer_name_list:
            overlap_params_number[layer_name] = overlap_params[(layer_name, 'kernel')].flatten().astype(bool).astype(int).sum()
            forward_params_number[layer_name] = (1 - current_grad_masks[(layer_name, 'kernel')]).flatten().astype(bool).astype(int).sum()
            total_overlap_params += overlap_params_number[layer_name]
            total_forward_params += forward_params_number[layer_name]
        for layer_name, overlap_params_number in overlap_params_number.items():
            # if each layer use different beta
            if FLAGS.use_adaptive_beta:
                current_beta[layer_name] = FLAGS.beta_lambda * get_beta(frozen_number=overlap_params_number, total_number=forward_params_number[layer_name])
            else:
                current_beta[layer_name] = FLAGS.default_beta
            wandb.log({
                f"overlap_params_number/{layer_name}": overlap_params_number,
                f"forward_params_number/{layer_name}": forward_params_number[layer_name],
                f"overlap_params_percentage/{layer_name}": (overlap_params_number / forward_params_number[layer_name]) * 100,
                'global_steps': total_env_steps
            })
        wandb.log({
            "total_overlap_params": total_overlap_params,
            "total_forward_params": total_forward_params,
            "total_overlap_params_ratio": total_overlap_params / total_forward_params,
            'global_steps': total_env_steps
        })

            # >>>>>>>>>>>>>>>>>>>>>> append the overlap >>>>>>>>>>>>>>>
        a = agent.actor.params
        a = unfreeze(a)
        a['overlap_params_dict'] = overlap_params_dict[task_idx]
        # current_beta = FLAGS.beta_lambda * get_beta(frozen_number=total_overlap_params, total_number=total_forward_params)
        # current_beta = 0.3

        
        for layer_name in layer_name_list:
            wandb.log({
                f"beta/{layer_name}": current_beta[layer_name],
                'global_steps': total_env_steps
            })

        beta_list = []
        for layer_name in layer_name_list:
            beta_list.append(current_beta[layer_name])
        beta_dict[task_idx] = beta_list
        a['beta'] = beta_list
        if FLAGS.multi_head:
            a['mean_layer']['kernel'] = head_dict[task_idx]
        a = FrozenDict(a)
        agent.actor = agent.actor.replace(params=a)

        wandb.log({
            'beta': a['beta'],
            'global_steps': total_env_steps
        })
            # <<<<<<<<<<<<<<<<<<<<<< append the over lap <<<<<<<<<<<<<<


        # >>>>>>>>>>>>>>>>>>>> log parameters situations <<<<<<<<<<<<<<<
        
        if task_idx > 0 and FLAGS.rnd_explore:
            '''
            (Optional) Rand policy distillation for better exploration in the initial stage
            '''
            for i in range(FLAGS.distill_steps):
                batch = replay_buffer.sample(FLAGS.batch_size)
                distill_info = agent.rand_net_distill(task_idx, batch)
                
                if i % (FLAGS.distill_steps // 10) == 0:
                    print(i, distill_info)
            # reset actor's optimizer
            agent.reset_actor_optimizer()
        
        # >>>>>>>>>>>>>>>>>>>> store the parameter before learning the new task >>>>>>>>>>>>>>>
        temp_params = agent.actor.params.copy() # NOTE: store the parameter before learning the new task
        # <<<<<<<<<<<<<<<<<<<< store the parameter before learning the new task <<<<<<<<<<<<<<<


        # set continual world environment
        env = get_single_env(
            dict_task['task'], FLAGS.seed, randomization=FLAGS.env_type, 
            normalize_reward=FLAGS.normalize_reward
        )
        # reset replay buffer
        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, FLAGS.buffer_size or FLAGS.max_step
        )
        evaluation_buffer = ReplayBuffer(
            env.observation_space, env.action_space, FLAGS.evaluation_batch_size
        )
        # reset scheduler
        schedule = itertools.cycle([False]*FLAGS.theta_step + [True]*FLAGS.alpha_step)
        # reset environment
        observation, done = env.reset(), False

        if FLAGS.use_quick_experiments and task_idx < 4:
            max_steps = FLAGS.first_four_task_max_steps
        else:
            max_steps = FLAGS.max_step
        
        for idx in range(max_steps):
            if idx < FLAGS.start_training:
                # initial exploration strategy proposed in ClonEX-SAC
                if task_idx == 0:
                    action = env.action_space.sample()
                else:
                    # uniform-previous strategy
                    mask_id = np.random.choice(task_idx)
                    action = agent.sample_actions(observation[np.newaxis], mask_id)
                    action = np.asarray(action, dtype=np.float32).flatten()
                
                # default initial exploration strategy
                # action = env.action_space.sample()
            else:
                action = agent.sample_actions(observation[np.newaxis], task_idx)
                action = np.asarray(action, dtype=np.float32).flatten()
                
            next_observation, reward, done, info = env.step(action)
            # counting total environment step
            total_env_steps += 1

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            # only for meta-world
            assert mask == 1.0

            replay_buffer.insert(
                observation, action, reward, mask, float(done), next_observation
            )
            evaluation_buffer.insert(
                observation, action, reward, mask, float(done), next_observation
            )
            # CRUCIAL step easy to overlook
            observation = next_observation

            if done:
                # EPISODIC ending
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})

            if (idx >= FLAGS.start_training) and (idx % FLAGS.updates_per_step == 0):
                for _ in range(FLAGS.updates_per_step):
                    batch = replay_buffer.sample(FLAGS.batch_size)
                    update_info = agent.update(task_idx, batch, next(schedule))
                if idx % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f'training/{k}': v, 'global_steps': total_env_steps})

            if idx % FLAGS.eval_interval == 0:
                if FLAGS.multi_head:
                    head_dict[task_idx] = unfreeze(agent.actor.params['mean_layer']['kernel'].copy())
                    eval_stats = evaluate_cl(agent, eval_envs, FLAGS.eval_episodes, current_task_id=task_idx, overlap_param=overlap_params_dict, beta_dict=beta_dict, multi_head=head_dict)
                else:
                    eval_stats = evaluate_cl(agent, eval_envs, FLAGS.eval_episodes, current_task_id=task_idx, overlap_param=overlap_params_dict, beta_dict=beta_dict)

                for k, v in eval_stats.items():
                    wandb.log({f'evaluation/{k}': v, 'global_steps': total_env_steps})

                # Update the log with collected data
                eval_stats['cl_method'] = algo
                eval_stats['x'] = total_env_steps
                eval_stats['steps_per_task'] = FLAGS.max_step
                log.update(eval_stats)
            # >>>>>>>>>>>>>>>>>>>> calculate the layer sensitivity <<<<<<<<<<<<<<<
            if FLAGS.use_input_sensitive:
                if idx % FLAGS.calculate_layer_sensitivity_interval == 0 and idx > 0 and idx < FLAGS.stop_reset_after_steps:
                    batch = evaluation_buffer.sample(FLAGS.evaluation_batch_size)
                    temp_observations = jnp.array(batch.observations)
                    agent.actor_with_intermediate = agent.actor_with_intermediate.replace(params=agent.actor.params)
                    delta_y, info = calculate_layer_neuron(temp_observations, agent.actor_with_intermediate, task_idx)
                    layer_neuron_difference = get_each_layer_neuron_difference(delta_y,info)
                    each_layer_reset_indices = get_each_layer_reset_indices(layer_neuron_difference, threshold=FLAGS.layer_neuron_threshold) # this function will indicate the indices of neurons to reset of each layer
                    total_reset_neurons = 0
                    total_neurons = 0
                    for layer_name, indices in each_layer_reset_indices.items(): # log some from neuron view
                        layer_total_neurons = info['masks'][layer_name][0].sum()
                        reset_percentage = (len(indices) / layer_total_neurons) * 100
                        total_reset_neurons += len(indices)
                        total_neurons += layer_total_neurons
                        wandb.log({
                            f"neuron_reset_indices/{layer_name}": len(indices),
                            f"neuron_reset_percentage/{layer_name}": reset_percentage,
                            'global_steps': total_env_steps
                        })
                    
                    total_reset_percentage = (total_reset_neurons / total_neurons) * 100
                    wandb.log({
                        "total_neuron_reset_indices": total_reset_neurons,
                        "total_neuron_reset_percentage": total_reset_percentage,
                        'global_steps': total_env_steps
                    })
                    available_indices = get_available_indices(info)
                    reset_params = reset_params_four_layers(agent.actor, temp_params, each_layer_reset_indices, available_indices)
                    # log reset params numbers
                    reset_diff = tree_map(lambda x, y: x - y, reset_params, agent.actor.params)
                    layer_name_list = ['backbones_0', 'backbones_1', 'backbones_2', 'backbones_3', 'mean_layer']
                    total_reset_params = 0
                    total_params = 0
                    for i, layer_name in enumerate(layer_name_list): # log some from parameter view
                        if layer_name == 'backbones_0':
                            total_params_to_reset = 12 * each_layer_reset_indices[layer_name].size # which will invlove some frozen parameters
                            pre_number = available_indices[layer_name].size
                        else:
                            total_params_to_reset = pre_number * each_layer_reset_indices[layer_name].size
                            pre_number = available_indices[layer_name].size
                        reset_params_number = reset_diff[layer_name]['kernel'].flatten().astype(bool).astype(int).sum()
                        total_reset_params += reset_params_number
                        total_params += total_params_to_reset
                        wandb.log({
                            f"reset_params/{layer_name}_total": total_params_to_reset,
                            f"reset_params/{layer_name}_reset_number": reset_params_number,
                            f"reset_params/{layer_name}_reset_percentage": (reset_params_number / total_params_to_reset) * 100,
                            'global_steps': total_env_steps
                        })
                    
                    # Log the average reset parameters and total reset numbers
                    wandb.log({
                        "reset_params/total_reset_number": total_reset_params,
                        "reset_params/total_params": total_params,
                        "reset_params/total_reset_percentage": (total_reset_params / total_params) * 100,
                        'global_steps': total_env_steps
                    })

                        
                    agent.actor = agent.actor.replace(params=reset_params)
                    agent.reset_actor_optimizer()
            # <<<<<<<<<<<<<<<<<<<< calculate the layer sensitivity <<<<<<<<<<<<<<<
    
        '''
        Updating miscellaneous things
        '''
        print('End of the current task')
        dict_stats = agent.end_task(task_idx, save_policy_dir, save_dict_dir)

        # >>>>>>>>>>>>>>>>>>>> restore the agent and cumul_masks and grad_masks >>>>>>>>>>>>>>>
        temp_params = agent.actor.params.copy()
        temp_params = unfreeze(temp_params)
        temp_params['overlap_params_dict'] = None
        if FLAGS.is_store_everything:
            store_folder_name = "restore_ckp/" + wandb.run.name
            if FLAGS.multi_head:
                head_dict[task_idx] = temp_params['mean_layer']['kernel']
            temp_params = FrozenDict(temp_params)
            agent.actor = agent.actor.replace(params=temp_params)
            agent.actor.save(f'{store_folder_name}/{FLAGS.seed}/{task_idx}/actor.pkl')
            
            # # Save cumul_masks
            with open(f'{store_folder_name}/{FLAGS.seed}/{task_idx}/cumul_masks.pkl', 'wb') as f:
                pickle.dump(agent.cumul_masks, f)
            
            with open(f'{store_folder_name}/{FLAGS.seed}/{task_idx}/param_masks.pkl', 'wb') as f:
                pickle.dump(agent.param_masks, f)
            
            # restore buffer
            with open(f'{store_folder_name}/{FLAGS.seed}/{task_idx}/replay_buffer.pkl', 'wb') as f:
                pickle.dump(replay_buffer, f)
            
            with open(f'{store_folder_name}/{FLAGS.seed}/{task_idx}/overlap_param_dict.pkl', 'wb') as f:
                pickle.dump(overlap_params_dict, f)
            
            with open(f'{store_folder_name}/{FLAGS.seed}/{task_idx}/beta_dict.pkl', 'wb') as f:
                pickle.dump(beta_dict, f)
        # <<<<<<<<<<<<<<<<<<<< restore the agent and cumul_masks and grad_masks <<<<<<<<<<<<<<<

    # save log data
    log.save()
    np.save(f'{wandb.run.dir}/dict_stats.npy', dict_stats)

def calculate_layer_neuron(observations, actor, task_id):
    info, intermediate = actor(observations, jnp.array([task_id]))
    # noise = jax.random.normal(jax.random.PRNGKey(0), shape=(observations.shape[0], 12))
    noise = observations.mean(axis=0) * 0.01
    # Add the noise to the observations
    perturbed_observations = observations.at[:, :12].add(noise)
    info_, perturbed_intermediate = actor(perturbed_observations, jnp.array([task_id]))
    delta_y = tree_map(lambda x, y: jnp.abs(x - y), intermediate['intermediates'], perturbed_intermediate['intermediates'])
    info[1]['masks']['mean_layer'] = jnp.ones(4)
    return delta_y, info[1]

def get_each_layer_neuron_difference(delta_y, info):
    layer_name = ['backbones_0', 'backbones_1', 'backbones_2', 'backbones_3']
    output = {}
    for layer_name in layer_name:
        denominator = tree_map(lambda x, y: x * y, delta_y[layer_name]['__call__'][0], info['masks'][layer_name])
        denominator =  denominator.mean(axis=0).sum() / info['masks'][layer_name][0].sum()
        molecule = delta_y[layer_name]['__call__'][0].mean(axis=0)
        molecule = jnp.where(info['masks'][layer_name][0] == 1, molecule, 0)
        # Drop the values which are 0 from molecule
        molecule = molecule[molecule != 0]
        output[layer_name] = molecule / denominator

    # mean layer
    layer_name = 'mean_layer'
    molecule = delta_y[layer_name]['__call__'][0].mean(axis=0)
    denominator = molecule.sum() / 4
    output[layer_name] = molecule / denominator

    return output

def get_each_layer_reset_indices(neuron_difference, threshold=0.01):
    output = {}
    for layer_name in ['backbones_0', 'backbones_1', 'backbones_2', 'backbones_3', 'mean_layer']:
        temp = []
        for i in range(neuron_difference[layer_name].shape[0]):
            if neuron_difference[layer_name][i] < threshold:
                temp.append(i)
        output[layer_name] = jnp.array(temp)
    return output

def get_available_indices(info):
    output = {}
    for layer_name in ['backbones_0', 'backbones_1', 'backbones_2', 'backbones_3']:
        flag = info['masks'][layer_name][0]
        available_indices = jnp.where(flag == 1)[0]
        output[layer_name] = available_indices
    # mean layer
    output['mean_layer'] = jnp.arange(4)
    return output

def get_new_params_by_layer(actor, temp_params, reset_indices, layer_name, available_indices):
    reset_indices = available_indices[layer_name][reset_indices[layer_name]]
    flag = jnp.zeros(actor.params[layer_name]['kernel'].T.shape)
    flag = flag.at[reset_indices].set(1)
    params = tree_map(lambda x, y, flag: (1 - flag) * x + flag * y, actor.params[layer_name]['kernel'].T, temp_params[layer_name]['kernel'].T, flag)
    params = params.T
    return params

def reset_params_four_layers(actor, temp_params, reset_indices, available_indices):
    # new_params = unfreeze(actor.params)
    new_params = unfreeze(temp_params)
    layer_name_list = ['backbones_0', 'backbones_1', 'backbones_2', 'backbones_3', 'mean_layer']
    for layer_name in layer_name_list:
        if reset_indices[layer_name].size > 0:
            new_layer_params = get_new_params_by_layer(actor, temp_params, reset_indices, layer_name, available_indices)
            new_params[layer_name]['kernel'] = new_layer_params
    return FrozenDict(new_params)

if __name__ == '__main__':
    app.run(main)
