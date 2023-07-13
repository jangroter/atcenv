"""
Example
"""

import numpy as np
import tracemalloc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

# choose between 'masac_transformer' or 'masac'
RLMETHOD = 'masac_transformer'

SCENARIO_NAME = 'test_scenario_high_density/'
NUM_SCENARIOS = 100
TEST_FREQUENCY = 25

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def run_episode(env, number_of_aircraft, test, tot_rew_list, conf_list, drift_array, index=0):
    if not test:
        obs = env.reset(number_of_aircraft)
    else:
        obs = env.load_scenario(SCENARIO_NAME,index,number_of_aircraft)
    env.close()
    for obs_i in obs:
        RL.normalizeState(obs_i, env.max_speed, env.min_speed)
    
    if RLMETHOD == 'masac_c_transformer':
        qobs = env.obs_relative()
        for qobs_i in qobs:
            RL.qnormalizeState(qobs_i, env.max_speed, env.min_speed)
    
    done = False
    number_steps_until_done = 0
    number_conflicts = 0
    tot_rew = 0

    # execute one episode
    while not done:

        actions = RL.do_step(obs,env.max_speed, env.min_speed, test=test)
        obs0 = copy.deepcopy(obs)
        obs, rew, done_t, done_e, info = env.step(actions)

        for obs_i in obs:
            RL.normalizeState(obs_i, env.max_speed, env.min_speed)

        if RLMETHOD == 'masac_c_transformer':
            qobs0 = copy.deepcopy(qobs)
            qobs = env.obs_relative()
            for qobs_i in qobs:
                RL.qnormalizeState(qobs_i, env.max_speed, env.min_speed)
        
        if done_t or done_e:
            done = True

        tot_rew += rew

        if len(env.done) == 0 and not test:
            if RLMETHOD == 'masac_c_transformer':
                RL.setResult(episode_name, obs0, qobs0, obs, qobs, sum(rew), actions, done_e)
            elif RLMETHOD == 'masac':
                RL.setResult(episode_name, obs0, obs, sum(rew), actions, done_e)
            else:
                RL.setResult(episode_name, obs0, obs, rew, actions, done_e)

        if e%25 == 0 and index == 0:
            try:
                env.render()
                time.sleep(0.02)
            except:
                pass

        number_steps_until_done += 1
        number_conflicts += len(env.conflicts)            

    tot_rew_list.append(sum(tot_rew)/number_of_aircraft)
    conf_list.append(number_conflicts)       
    
    env.drift_array.append(np.mean(np.array(env.drift_array_eps)))  
    drift_array.append(np.mean(np.array(env.drift_array_eps)))
    env.drift_array_eps = []
    return tot_rew_list, conf_list, drift_array

def plot_figures(tot_rew_list):
    fig, ax = plt.subplots()
    ax.plot(RL.qf1_lossarr, label='qf1')
    ax.plot(RL.qf2_lossarr, label='qf2')
    ax.plot(moving_average(RL.qf2_lossarr,500))
    ax.set_yscale('log')
    fig.savefig('qloss.png')
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(tot_rew_list, label='reward')
    ax.plot(moving_average(tot_rew_list,500))
    fig.savefig('reward.png')
    plt.close(fig)
    # RL.actor.print = True

if __name__ == "__main__":
    import random
    random.seed(52)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment
    import time
    from tqdm import tqdm

    # RL model
    import atcenv.TempConfig as tc
    if RLMETHOD == 'masac':
        from atcenv.MASAC.masac_agent import MaSacAgent
        obs_type = 'relative'
    elif RLMETHOD == 'masac_transformer':
        from atcenv.MASAC_transform.masac_agent import MaSacAgent
        obs_type = 'absolute'
    elif RLMETHOD == 'masac_transformer_log_critic':
        from atcenv.MASAC_transform_log_critic.masac_agent import MaSacAgent
        obs_type = 'absolute'
    elif RLMETHOD == 'masac_c_transformer':
        from atcenv.MASAC_C_transform.masac_agent import MaSacAgent
        obs_type = 'absolute'
    elif RLMETHOD == 'masac_transformer_rel_action':
        from atcenv.MASAC_transform_rel_action.masac_agent import MaSacAgent
        obs_type = 'absolute'
    elif RLMETHOD == 'masac_transformer_intent':
        from atcenv.MASAC_transform_intent.masac_agent import MaSacAgent
        obs_type = 'absolute'
    else:
        raise Exception("Choose between 'masac' and 'masac_transformer'.")
    
    import copy

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()
    args.env.obs = obs_type
    #tracemalloc.start()

    # init environment
    env = Environment(**vars(args.env))

    #RL = DDPG()
    RL = MaSacAgent()

    best_score = -999

    load_models = False
    test = False

    if load_models:
        RL.load_models()
    # increase number of flights
    tot_rew_list = []
    conf_list = []
    drift_array = []

    test_rew_list = []
    test_conf_list = []
    test_drift_list = []
    # run episodes
    state_list = []
    for e in tqdm(range(args.episodes)):   
        episode_name = "EPISODE_" + str(e) 
        number_of_aircraft = 15 
        
        if e % TEST_FREQUENCY == 5:
            test = True
            RL.set_test(True)

        if not test:
            tot_rew_list, conf_list, drift_array = run_episode(env, number_of_aircraft, test, tot_rew_list, conf_list, drift_array)
            np.savetxt('conflict.csv',np.array(conf_list))
            np.savetxt('drift.csv',np.array(drift_array))
            np.savetxt('reward.csv',np.array(tot_rew_list))
            print(episode_name,'ended in with', np.mean(np.array(conf_list[-100:])), 'conflicts (rolling av100), reward (rolling av100)=', np.mean(np.array(tot_rew_list[-100:])),', best score: ', best_score)    
            env.close()   

        
        else: 
            print(f'Now testing, episode: {e}')
            test_rew = []
            test_conf = []
            test_drift = []

            for i in range(0,NUM_SCENARIOS):
                test_rew, test_conf, test_drift = run_episode(env, number_of_aircraft, test, test_rew, test_conf, test_drift, i)
                env.close()
            
            test_rew_list.append(list(test_rew))
            test_conf_list.append(list(test_conf))
            test_drift_list.append(list(test_drift))

            np.savetxt('test_drift.csv',np.array(test_drift_list))
            np.savetxt('test_reward.csv',np.array(test_rew_list))
            np.savetxt('test_conflict.csv',np.array(test_conf_list))

            score = np.mean(test_rew)
            if score > best_score:
                print(f'beat my pb: old score was {best_score}, now have {score}!!!')
                best_score = score
                RL.save_models()

            test = False
            RL.set_test(False)
        
        if e%25 == 0 and e > 0:
            plot_figures(tot_rew_list)



        
