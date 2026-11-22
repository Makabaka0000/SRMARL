from MAPPOCVaR_Gussian import MAPPOCVaRG
from common.utils import agg_double_list, copy_file_ppo, init_dir
import sys
sys.path.append(r"\highway-env")
import gym
import numpy as np
import matplotlib.pyplot as plt
import highway_env
import argparse
import configparser
import os
from datetime import datetime,timezone
import csv
from tensorboardX import SummaryWriter
 

def parse_args():
    """
    Description for this experiment:
        + easy: globalR
        + seed = 0
    """
    default_base_dir =  r"MARL"
    default_config_dir =  r"configs_ppo.ini"
    default_model_dir =  r"default_model_dirs"
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using mappo'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False,
                        default='train', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--model-dir', type=str, required=False,
                        default=default_model_dir, help="pretrained model path")
    

    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0,35,1)]),
                        help="random seeds for evaluation, split by ,")

    args = parser.parse_args()
    return args


def train(args):
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # create an experiment folder
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = base_dir + now
    dirs = init_dir(output_dir)
    copy_file_ppo(dirs['configs'])

    if os.path.exists(args.model_dir):
        model_dir = args.model_dir
    else:
        model_dir = dirs['models']

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    TARGET_UPDATE_STEPS = config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS')
    TARGET_TAU = config.getfloat('MODEL_CONFIG', 'TARGET_TAU')

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    MAX_EPISODES = config.getint('TRAIN_CONFIG', 'MAX_EPISODES')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    EVAL_INTERVAL = config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL')
    EVAL_EPISODES = config.getint('TRAIN_CONFIG', 'EVAL_EPISODES')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')

    # init env
    env = gym.make('merge-multi-agent-v0')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    env.config['safety_guarantee'] = config.getboolean('MODEL_CONFIG', 'safety_guarantee')
    assert env.T % ROLL_OUT_N_STEPS == 0

    env_eval = gym.make('merge-multi-agent-v0')
    env_eval.config['seed'] = config.getint('ENV_CONFIG', 'seed') + 1
    env_eval.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env_eval.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env_eval.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env_eval.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env_eval.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env_eval.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env_eval.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env_eval.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env_eval.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env_eval.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    env.config['safety_guarantee'] = config.getboolean('MODEL_CONFIG', 'safety_guarantee')
    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    mappo = MAPPOCVaRG(env=env, state_dim=state_dim, action_dim=action_dim,writer=writer,
                  total_training_episodes=MAX_EPISODES,memory_capacity=MEMORY_CAPACITY,
                  batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                  roll_out_n_steps=ROLL_OUT_N_STEPS,
                  actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                  actor_lr=actor_lr, critic_lr=critic_lr, reward_scale=reward_scale,
                  target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
                  reward_gamma=reward_gamma, reward_type=reward_type,
                  max_grad_norm=MAX_GRAD_NORM, test_seeds=test_seeds,
                  episodes_before_train=EPISODES_BEFORE_TRAIN, traffic_density=traffic_density
                  )

    # load the model if exist
    mappo.load(model_dir, train_mode=True)
    env.seed = env.config['seed']
    env.unwrapped.seed = env.config['seed']
    target_dir=os.path.join(output_dir,"eval_logs")
    os.makedirs(target_dir,exist_ok=True)
    target_file=os.path.join(target_dir,"ppo_val_rewards.txt")

    eval_rewards = []
    eval_costs = []



    while mappo.n_episodes < MAX_EPISODES:
        mappo.interact()
        print(f"Episode: {mappo.n_episodes}, Step: {mappo.n_steps}, Done: {mappo.episode_done}")
        if mappo.n_episodes >= EPISODES_BEFORE_TRAIN:
            mappo.train()
        if mappo.episode_done and ((mappo.n_episodes + 1) % EVAL_INTERVAL == 0):
            rewards,costs, _, _, _ = mappo.evaluation(env_eval, dirs['train_videos'], EVAL_EPISODES)
            rewards_mu, _ = agg_double_list(rewards)
            costs_mu, costs_std = agg_double_list(costs)
            print("Episode %d, Average Reward %.2f, Average Cost %.2f" % (mappo.n_episodes +  1, rewards_mu,costs_mu))
            eval_rewards.append(rewards_mu)
            eval_costs.append(costs_mu) 
            ###tensorboard记录
            writer.add_scalar("eval/average_reward", rewards_mu, mappo.n_episodes + 1)
            writer.add_scalar("eval/average_cost", costs_mu, mappo.n_episodes + 1)
            mappo.save(dirs['models'], mappo.n_episodes + 1)

    np.savetxt(target_file, np.array(eval_rewards))
    np.savetxt(target_file.replace("rewards","costs"), np.array(eval_costs))
    # save the model
    mappo.save(dirs['models'], MAX_EPISODES + 2)
    writer.close()

    plt.figure()
    plt.plot(eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["MAPPO"])
    plt.show()


def evaluate(args):
    if os.path.exists(args.model_dir):
        model_dir = args.model_dir + '\\models\\'
    else:
        raise Exception("Sorry, no pretrained models")
    config_dir = args.model_dir + '\\configs\\configs_ppo.ini'
    config = configparser.ConfigParser()
    config.read(config_dir)

    video_dir = args.model_dir + '\\eval_videos'

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    TARGET_UPDATE_STEPS = config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS')
    TARGET_TAU = config.getfloat('MODEL_CONFIG', 'TARGET_TAU')

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')

    # init env
    env = gym.make('merge-multi-agent-v0')
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    env.config['safety_guarantee'] = config.getboolean('MODEL_CONFIG', 'safety_guarantee')

    assert env.T % ROLL_OUT_N_STEPS == 0
    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(',')]


    writer = None
    MAX_EPISODES=20000
    mappo = MAPPOCVaRG(env=env, state_dim=state_dim, action_dim=action_dim,writer=writer,
                  total_training_episodes=MAX_EPISODES,memory_capacity=MEMORY_CAPACITY,
                  batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                  roll_out_n_steps=ROLL_OUT_N_STEPS,
                  actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                  actor_lr=actor_lr, critic_lr=critic_lr, reward_scale=reward_scale,
                  target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
                  reward_gamma=reward_gamma, reward_type=reward_type,
                  max_grad_norm=MAX_GRAD_NORM, test_seeds=test_seeds,
                  episodes_before_train=EPISODES_BEFORE_TRAIN, traffic_density=traffic_density
                  )

    # load the model if exist
    mappo.load(model_dir, train_mode=False)
    rewards, costs,(vehicle_speed, vehicle_position), steps, avg_speeds,s_r,cr = mappo.evaluation(env, video_dir, len(seeds), is_train=False)
    rewards_mu, _ = agg_double_list(rewards)
    costs_mu, _ = agg_double_list(costs)
    print("Average Reward %.2f, Average Cost %.2f, crash ratio %.6f, Average speed %.2f, success_rate %.6f" % (rewards_mu,costs_mu,cr,np.mean(avg_speeds),s_r))

if __name__ == "__main__":
    args = parse_args()
    # train or eval
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
