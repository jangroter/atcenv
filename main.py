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

SCENARIO_NAME = 'test_scenario/'
NUM_SCENARIOS = 100
TEST_FREQUENCY = 100

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

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

    load_models = False
    test = False

    if load_models:
        RL.load_models()
    # increase number of flights
    tot_rew_list = []
    conf_list = []
    # run episodes
    state_list = []
    for e in tqdm(range(args.episodes)):   
        print('\n-----------------------------------------------------')   
        episode_name = "EPISODE_" + str(e) 
        number_of_aircraft = 10 

        # obs = env.reset(number_of_aircraft)
        obs = env.load_scenario(SCENARIO_NAME,1,number_of_aircraft)
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

            if len(env.done) == 0:
                if RLMETHOD == 'masac_c_transformer':
                    RL.setResult(episode_name, obs0, qobs0, obs, qobs, sum(rew), actions, done_e)
                elif RLMETHOD == 'masac':
                    RL.setResult(episode_name, obs0, obs, sum(rew), actions, done_e)
                else:
                    RL.setResult(episode_name, obs0, obs, rew, actions, done_e)

            if e%1 == 0:
                try:
                    env.render()
                    time.sleep(0.02)
                except:
                    pass

            number_steps_until_done += 1
            number_conflicts += len(env.conflicts)            

        tot_rew_list.append(sum(tot_rew)/number_of_aircraft)
        conf_list.append(number_conflicts)

        if e%25 == 0 and e > 10:
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
            # ax.set_ylim([0,10])
            fig.savefig('reward.png')
            plt.close(fig)
            RL.actor.print = True
        
        if e%100 == 0:
            RL.save_models()

        env.drift_array.append(np.mean(np.array(env.drift_array_eps)))
        env.drift_array_eps = []  
        np.savetxt('drift.csv',env.drift_array)
        np.savetxt('reward.csv',np.array(tot_rew_list))
        tc.dump_pickle(number_steps_until_done, 'results/save/numbersteps_' + episode_name)
        tc.dump_pickle(number_conflicts, 'results/save/numberconflicts_' + episode_name)
        print(f'Done aircraft: {len(env.done)}')  
        print(f'Done aircraft IDs: {env.done}')      

        print(episode_name,'ended in', number_steps_until_done, 'runs, with', np.mean(np.array(conf_list[-100:])), 'conflicts (rolling av100), reward (rolling av100)=', np.mean(np.array(tot_rew_list[-100:])))       

        env.close()
