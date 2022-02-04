import os
import pickle
from Two_step import *

# -------------------------------------------------------------------------------------
# Script functions
# -------------------------------------------------------------------------------------

def save_data():
    try:
        data_struct = {'agents_r'         : agents_r,
                       'agents_o'         : agents_o,
                       'matched_params_r' : matched_params_r,
                       'matched_params_o' : matched_params_o,
                       'sessions_r'       : sessions_r, 
                       'sessions_o'       : sessions_o, 
                       'grid_sims_r'      : grid_sims_r,
                       'data_r'           : data_r,
                       'data_o'           : data_o,
                       'performance_r'    : performance_r, 
                       'performance_o'    : performance_o,
                       'performance_o_gs' : performance_o_gs, 
                       'performance_tr'   : performance_tr,
                       'likelihood_data_r': likelihood_data_r,
                       'likelihood_data_o': likelihood_data_o}
        with open(os.path.join(data_dir,'data_struct.pkl'), 'wb') as data_file:
            pickle.dump(data_struct, data_file)
        print('Data saved to file: data_struct.pkl')
    except IOError:
        print('Data not saved as data directory does not exist.')

def plots():

    # Figure 1:
    pl.sessions_data_plot(data_o['Q(1)'], 1)
    pl.sessions_data_plot(data_o['MB']  , 2)
    pl.sessions_data_plot(data_r['Q(1)'], 3)
    pl.sessions_data_plot(data_r['MB']  , 4)

    # Figure 2:
    pl.action_values_plot(sessions_r['Q(1)'], agents_r['Q(1)'],  5)
    pl.predictor_correlations(sessions_r['Q(1)'],lr.CCOTI_model, 6)

    # Figure 3:
    pl.sessions_data_plot(data_r['Q(0)'], 7)
    pl.sessions_data_plot(data_r['RC']  , 8)
    pl.sessions_data_plot(data_r['LS']  , 9)

    # Figure 4:
    pl.performance_plot(performance_r, 10)
    pl.performance_plot(performance_o, 11)

    # Figure 6:
    pl.likelihood_comparison_plot(likelihood_data_r, 12)
    pl.likelihood_comparison_plot(likelihood_data_o, 13)

    # Figure S1.
    pl.multi_agent_log_reg_surface_plot(grid_sims_r, 14)

    # Figure S2.
    pl.sessions_data_plot(data_r['Q(1)_w'],      15)
    pl.fit_plot(data_r['Q(1)_w']['CcCOTI_fits'], 16)
    pl.sessions_data_plot(data_r['Q(1)_f']     , 17)
    pl.sessions_data_plot(data_r['Q(1)_n']     , 18)

    # Figure S3:
    pl.sessions_data_plot(data_o['Q(0)'], 19)
    pl.sessions_data_plot(data_o['RC']  , 20)
    pl.sessions_data_plot(data_o['LS']  , 21)

    # Figure S4
    pl.sessions_data_plot(data_r['Q(.25)'], 22)
    pl.sessions_data_plot(data_r['Q(.5)' ], 23)
    pl.sessions_data_plot(data_r['Q(.75)'], 24)

    # Figure S5.
    pl.sessions_data_plot(data_r['MB_tl0.1'], 25)
    pl.sessions_data_plot(data_r['MB_tl0.5'], 26)
    pl.sessions_data_plot(data_r['MB_tl0.9'], 27)

    # Figure S6
    pl.sessions_data_plot(data_r['MB_pv0.4'] ,  28)
    pl.sessions_data_plot(data_r['Q(1)_pv0.4'], 29)

    # Figure S7
    pl.likelihood_comparison_plot(likelihood_data_o, 30, use_BIC = True)

# -------------------------------------------------------------------------------------
# Script 
# -------------------------------------------------------------------------------------

load_data = True     # Set to False to re-simulate rather than load data from file.
plot_figures = True  # Set to false to supress plotting.

# Simulation parameters

n_sessions = 10     # Number of sessions per agent.
n_trials   = 10000  # Number of trials per session.

data_dir = os.path.join('..', 'data')

task_b = ts.Two_step()                                         # Task with blocked reward probabilities.
task_n = ts.Two_step(rew_gen = 'fixed', probs = [0.5, 0.5])    # Task with fixed neutral reward probabilities.
task_f = ts.Two_step(rew_gen = 'fixed', probs = [0.8, 0.2])    # Task with fixed 0.8, 0.2 reward probabilities.
task_w = ts.Two_step(rew_gen = 'walks')                        # Task with random walk reward probabilities.
task_o = ts.Orig_two_step()                                    # Task version used in Daw et al. 2011
task_tr = ts.Two_step(rew_gen='trans_rev', probs = [0.8, 0.2]) # Task version with reversals in the transition matrix.

if load_data:

    try:  # Load data from file if available.
        with open(os.path.join(data_dir,'data_struct.pkl'), 'rb') as data_file:
            data_struct = pickle.load(data_file)
        agents_r          = data_struct['agents_r']
        agents_o          = data_struct['agents_o']
        matched_params_r  = data_struct['matched_params_r']
        matched_params_o  = data_struct['matched_params_o']
        sessions_r        = data_struct['sessions_r']
        sessions_o        = data_struct['sessions_o']
        grid_sims_r       = data_struct['grid_sims_r']
        data_r            = data_struct['data_r']
        data_o            = data_struct['data_o']
        performance_r     = data_struct['performance_r']
        performance_o     = data_struct['performance_o']
        performance_o_gs  = data_struct['performance_o_gs'] 
        performance_tr    = data_struct['performance_tr']
        likelihood_data_r = data_struct['likelihood_data_r']
        likelihood_data_o = data_struct['likelihood_data_o']
        # Set agent parameters to matched parameters.
        for a_type in matched_params_r.keys():
            agents_r[a_type].params = matched_params_r[a_type] 
        for a_type in matched_params_o.keys():
            agents_o[a_type].params = matched_params_o[a_type] 
        print('Data loaded from file.')

    except IOError: # Otherwise simulate and analyse data.
        load_data = False

if not load_data:
    #Create agents.

    agents_r = {'Q(1)'   : rl.Q1(),   # Agents for use on reduced version of task.
                'Q(0)'   : rl.Q0(),
                'Q(.25)' : rl.Q_lambda(0.25),
                'Q(.5)'  : rl.Q_lambda(0.5),
                'Q(.75)' : rl.Q_lambda(0.75),
                'MB'     : rl.Model_based(),
                'RC'     : rl.Reward_as_cue(),
                'RF'     : rl.RAC_fixed(),
                'LS'     : rl.Latent_state()}

    agents_o = {'Q(1)'   : rl.Q1_orig(),   # Agents for use on original version of task.
                'Q(0)'   : rl.Q0_orig(),
                'MB'     : rl.Model_based_orig(),
                'RC'     : rl.Reward_as_cue_orig(),
                'RF'     : rl.RAC_fixed_orig(),
                'LS'     : rl.Latent_state_orig(),
                'RND'    : rl.Random_first_step_orig()}

    # Match parameters to give similar average behaviour.

    matched_params_r = mf.match_agent_parameters(agents_r, task_b, n_ses = n_sessions, n_trials = n_trials)
    matched_params_o = mf.match_agent_parameters(agents_o, task_o, n_ses = n_sessions, n_trials = n_trials)

    # Additional agents who inherit some params from other agents.
    agents_r['Q(1)_pv0.2'] = rl.Q1_prsv(params = agents_r['Q(1)'].params + [0.2])
    agents_r['Q(1)_pv0.4'] = rl.Q1_prsv(params = agents_r['Q(1)'].params + [0.4])
    agents_r['MB_pv0.2'  ] = rl.MB_prsv(params = agents_r['MB'  ].params + [0.2])
    agents_r['MB_pv0.4'  ] = rl.MB_prsv(params = agents_r['MB'  ].params + [0.4])    
    agents_r['MB_tl0.1'] = rl.MB_trans_learn(params = agents_r['MB'].params + [0.1])
    agents_r['MB_tl0.5'] = rl.MB_trans_learn(params = agents_r['MB'].params + [0.5])
    agents_r['MB_tl0.9'] = rl.MB_trans_learn(params = agents_r['MB'].params + [0.9])

    # Simulate sessions.

    # Simulations on reduced task with blockwise reward distributions.

    sessions_r = {a_type: sm.simulated_sessions(task_b, agents_r[a_type], n_sessions, n_trials)
                   for a_type in agents_r.keys()}  

    # # Simulations on other reduced task versions

    sessions_r['Q(1)_n'] =  sm.simulated_sessions(task_n, agents_r['Q(1)'  ], n_sessions, n_trials) 
    sessions_r['Q(1)_f'] =  sm.simulated_sessions(task_f, agents_r['Q(1)'  ], n_sessions, n_trials)
    sessions_r['Q(1)_w'] =  sm.simulated_sessions(task_w, agents_r['Q(1)'  ], n_sessions, n_trials)
    sessions_r['MB_n'  ] =  sm.simulated_sessions(task_n, agents_r['MB'    ], n_sessions, n_trials)

    # Simulations over grid of parameter values.

    grid_sims_r = {a_type: sm.param_grid_sim(task_b, agents_r[a_type], n_sessions, n_trials)
                  for a_type in ['Q(1)', 'Q(0)', 'MB', 'RC', 'LS']}

    # Simulations on original task version.

    sessions_o = {a_type: sm.simulated_sessions(task_o, agents_o[a_type], n_sessions, n_trials)
                  for a_type in agents_o.keys()}  

    # Analyse sessions.

    data_r = {s_type: pl.analyse_sessions(sessions_r[s_type]) for s_type in sessions_r.keys()}
    data_o = {s_type: pl.analyse_sessions(sessions_o[s_type]) for s_type in sessions_o.keys()}

    data_r['Q(1)_w']['CcCOTI_fits'] = mf.fit_sessions(sessions_r['Q(1)_w'], lr.CcCOTI_model)

    performance_r    = mf.optimise_agents_parameters   (agents_r, task_b , n_trials, n_sessions)
    performance_o    = mf.optimise_agents_parameters   (agents_o, task_o , n_trials, n_sessions)
    performance_o_gs = mf.optimise_agents_parameters_gs(agents_o, task_o , n_trials, n_sessions)
    performance_tr   = mf.optimise_agents_parameters   (agents_r, task_tr, n_trials, n_sessions,
                            agents_to_compare =  ['Q(1)', 'Q(0)', 'MB_tl0.5', 'RF', 'LS'])

    likelihood_data_r = pl.likelihood_comparison(sessions_r, agents_r)
    likelihood_data_o = pl.likelihood_comparison(sessions_o, agents_o)

if plot_figures:
    plots()

    # save_data()

