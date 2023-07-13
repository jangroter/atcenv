"""
Simulation script using the MVP conflict resolution algorithm for controlling the aircraft
"""

import numpy as np
import tracemalloc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

SCENARIO_NAME = 'test_scenario_high_density/'
NUM_SCENARIOS = 100
TEST_FREQUENCY = 25

T_LOOKAHEAD = 3000
NUM_AC = 15
MARGIN = 1.05

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def mvp(env, conflicts):
    x_array = np.zeros((NUM_AC,1))
    y_array = np.zeros((NUM_AC,1))
    vx_array = np.zeros((NUM_AC,1))
    vy_array = np.zeros((NUM_AC,1))

    I = np.eye(NUM_AC)

    for i, f in enumerate(env.flights):
        x_array[i] = f.position.x
        y_array[i] = f.position.y
        vx, vy = f.components
        vx_array[i] = vx 
        vy_array[i] = vy 
    
    # add additional dimension to array to allow transposing

    x_array = np.asmatrix(x_array)
    y_array = np.asmatrix(y_array)
    vx_array = np.asmatrix(vx_array)
    vy_array = np.asmatrix(vy_array)

    dx = x_array - x_array.T
    dy = y_array - y_array.T

    dist = np.sqrt(np.multiply(dx,dx) + np.multiply(dy,dy)) 

    dvx = vx_array - vx_array.T
    dvy = vy_array - vy_array.T

    dv2 = np.multiply(dvx, dvx) + np.multiply(dvy, dvy)
    dv2 = np.where(np.abs(dv2) < 1e-6, 1e-6, dv2)  # limit lower absolute value
    vrel = np.sqrt(dv2)

    tcpa = -(np.multiply(dvx, dx) + np.multiply(dvy, dy)) / dv2 + 1e9 * I

    # Calculate distance^2 at CPA (minimum distance^2)
    dcpa2 = np.abs(np.multiply(dist, dist) - np.multiply(np.multiply(tcpa, tcpa),  dv2))
    dcpa = np.sqrt(dcpa2)

    action = np.zeros((NUM_AC,2))

    for i in range(NUM_AC):
        delta_vx = 0
        delta_vy = 0
        for j in range(NUM_AC):
            # if tcpa[i,j]>0:
            #     print(f'dx is {dx[i,j]}, dy is {dy[i,j]}, and tcpa is {tcpa[i,j]}')
            #     env.render()
            #     input("Press Enter to continue...")
            if i != j:
                if (tcpa[i,j] < T_LOOKAHEAD and tcpa[i,j] > 0 and dcpa[i,j] < env.min_distance * MARGIN) or (dist[i,j] < env.min_distance * MARGIN):
                    dcpa_x = dx[i,j] + dvx[i,j]*tcpa[i,j]
                    dcpa_y = dy[i,j] + dvy[i,j]*tcpa[i,j]
                    
                    # Add aircraft 'j' to conflict list 
                    if j not in conflicts[i]:
                        conflicts[i].append(j)

                    # Compute horizontal intrusion
                    iH = env.min_distance - dcpa[i,j]
                    # print(dcpa[i,j])
                    # print(dist[i,j])
                    # Exception handlers for head-on conflicts
                    # This is done to prevent division by zero in the next step
                    if dcpa[i,j] <= 0.1:
                        dcpa[i,j] = 0.1
                        dcpa_x = dy[i,j] / dist[i,j] * dcpa[i,j]
                        dcpa_y = -dx[i,j] / dist[i,j] * dcpa[i,j]
                    
                    # If intruder is outside the ownship PZ, then apply extra factor
                    # to make sure that resolution does not graze IPZ
                    if env.min_distance < dist[i,j] and dcpa[i,j] < dist[i,j]:
                        # Compute the resolution velocity vector in horizontal direction.
                        # abs(tcpa) because it bcomes negative during intrusion.
                        erratum = np.cos(np.arcsin(env.min_distance / dist[i,j])-np.arcsin(dcpa[i,j] / dist[i,j]))
                        dv1 = ((env.min_distance / erratum - dcpa[i,j]) * dcpa_x) / (abs(tcpa[i,j]) * dcpa[i,j])
                        dv2 = ((env.min_distance / erratum - dcpa[i,j]) * dcpa_y) / (abs(tcpa[i,j]) * dcpa[i,j])
                    else:
                        dv1 = (iH * dcpa_x) / (abs(tcpa[i,j]) * dcpa[i,j])
                        dv2 = (iH * dcpa_y) / (abs(tcpa[i,j]) * dcpa[i,j])
                    
                    delta_vx -= dv1
                    delta_vy -= dv2

        new_vx = vx_array[i] + delta_vx
        new_vy = vy_array[i] + delta_vy

        oldtrack = (np.arctan2(vx_array[i],vy_array[i])*180/np.pi) % 360
        newtrack = (np.arctan2(new_vx,new_vy)*180/np.pi) % 360

        action[i,0] = np.deg2rad(oldtrack-newtrack)

        oldgs = np.sqrt(vx_array[i] * vx_array[i] + vy_array[i] * vy_array[i])
        newgs = np.sqrt(new_vx * new_vx + new_vy * new_vy)

        action[i,1] = oldgs-newgs

        for j in list(conflicts[i]):
            if tcpa[i,j] < 0:
                conflicts[i].remove(j)
        
        if not conflicts[i]:
            print('returning to path')
            action[i,0] = env.flights[i].drift
            action[i,1] = env.flights[i].optimal_airspeed

    return action, conflicts

def run_episode(env, number_of_aircraft, test, tot_rew_list, conf_list, drift_array, index=0):
    if not test:
        _ = env.reset(number_of_aircraft)
    else:
        _ = env.load_scenario(SCENARIO_NAME,index,number_of_aircraft)
    
    env.close()

    done = False
    number_steps_until_done = 0
    number_conflicts = 0
    tot_rew = 0

    conflicts = [[] for i in range(number_of_aircraft)]

    # execute one episode
    while not done:

        actions, conflicts = mvp(env,conflicts)   
        obs, rew, done_t, done_e, info = env.step(actions)
        
        if done_t or done_e:
            done = True

        tot_rew += rew

        if e%1== 0 and index%10== 0:
            try:
                env.render()
                time.sleep(0.01)
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

if __name__ == "__main__":
    import random
    random.seed(52)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment
    import time
    from tqdm import tqdm

    # RL model
    import atcenv.TempConfig as tc
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

    best_score = -999
    # init environment
    env = Environment(**vars(args.env))

    tot_rew_list = []
    conf_list = []
    drift_array = []

    test_rew_list = []
    test_conf_list = []
    test_drift_list = []

    test = True
    # run episodes
    state_list = []
    for e in tqdm(range(args.episodes)):   
        episode_name = "EPISODE_" + str(e) 
        number_of_aircraft = 15

        if not test:
            tot_rew_list, conf_list, drift_array = run_episode(env, number_of_aircraft, test, tot_rew_list, conf_list, drift_array)
            np.savetxt('conflict_mvp.csv',np.array(conf_list))
            np.savetxt('drift_mvp.csv',np.array(drift_array))
            np.savetxt('reward_mvp.csv',np.array(tot_rew_list))
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

            np.savetxt('test_drift_mvp.csv',np.array(test_drift_list))
            np.savetxt('test_reward_mvp.csv',np.array(test_rew_list))
            np.savetxt('test_conflict_mvp.csv',np.array(test_conf_list))

            score = np.mean(test_rew)
            print(score)
            if score > best_score:

                print(f'beat my pb: old score was {best_score}, now have {score}!!!')
                best_score = score




        