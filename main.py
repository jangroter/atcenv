"""
Example
"""

import numpy as np
import tracemalloc

if __name__ == "__main__":
    import random
    random.seed(52)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment
    import time
    from tqdm import tqdm

    # RL model
    import atcenv.TempConfig as tc
    from atcenv.MASAC_transform.masac_agent import MaSacAgent
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
        #snapshot1 = tracemalloc.take_snapshot()     
        episode_name = "EPISODE_" + str(e) 

        # reset environment
        # train with an increasing number of aircraft
        number_of_aircraft = 10 #min(int(e/500)+5,10)
        obs = env.reset(number_of_aircraft)
        for obs_i in obs:
            RL.normalizeState(obs_i, env.max_speed, env.min_speed)
        # set done status to false
        done = False

        # save how many steps it took for this episode to finish
        number_steps_until_done = 0
        # save how many conflics happened in eacj episode
        number_conflicts = 0
        tot_rew = 0
        # execute one episode
        while not done:
            #for obs_i in obs:
            # print(obs_i)
            actions = RL.do_step(obs,env.max_speed, env.min_speed, test=test)
                # actions.append((np.random.rand(2)-0.5)*2)
                #actions.append([0,0])

            obs0 = copy.deepcopy(obs)

            # perform step with dummy action
            obs, rew, done_t, done_e, info = env.step(actions)

            for obs_i in obs:
               RL.normalizeState(obs_i, env.max_speed, env.min_speed)

            if done_t or done_e:
                done = True

            #for obs_i in obs:
            #    state_list.append(obs_i)
            tot_rew += rew
            # train the RL model
            #for it_obs in range(len(obs)):
            #while len(obs) < len(obs0):
            #    obs.append( [0] * 14) # STATE_SIZE = 14
            if len(env.done) == 0:
                RL.setResult(episode_name, obs0, obs, sum(rew), actions, done_e)
                # print('obs0,',obs0[it_obs],'obs,',obs[it_obs],'done_e,', done_e)
            # comment render out for faster processing
            # if e%10 == 0:
            #     env.render()
                #time.sleep(0.01)
            number_steps_until_done += 1
            number_conflicts += len(env.conflicts)            
                
        if len(tot_rew_list) < 100:
            tot_rew_list.append(sum(tot_rew)/number_of_aircraft)
            conf_list.append(number_conflicts)
        else:
            tot_rew_list[e%100 -1] = sum(tot_rew)/number_of_aircraft
            conf_list[e%100 -1] = number_conflicts
        # save information
        # if not test:
        #     RL.learn() # train the model
        if e%100 == 0:
            RL.save_models()
        #RL.episode_end(episode_name)
        #np.savetxt('states.csv', state_list)
        tc.dump_pickle(number_steps_until_done, 'results/save/numbersteps_' + episode_name)
        tc.dump_pickle(number_conflicts, 'results/save/numberconflicts_' + episode_name)
        print(f'Done aircraft: {len(env.done)}')  
        print(f'Done aircraft IDs: {env.done}')      

        print(episode_name,'ended in', number_steps_until_done, 'runs, with', np.mean(np.array(conf_list)), 'conflicts (rolling av100), reward (rolling av100)=', np.mean(np.array(tot_rew_list))/20.)        
        #snapshot2 = tracemalloc.take_snapshot()
        #top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        #print("[ Top 10 differences ]")
        #for stat in top_stats[:10]:
        #    print(stat)
        # close rendering
        env.close()