import numpy as np
import random
import scipy.optimize as op
import math
from scipy.stats import sem, ttest_1samp, ttest_ind
from .logistic_regression import config_log_reg

# -------------------------------------------------------------------------------------
# Utility functions.
# -------------------------------------------------------------------------------------

def trans_UC(values_U, param_ranges):
    'Transform parameters from unconstrained to constrained space.'
    if param_ranges[0] == 'all_unc':
        return values_U
    values_T = []
    for value, rng in zip(values_U, param_ranges):
        if rng   == 'unit':  # Range: 0 - 1.
            if value < -16.:
                value = -16.
            values_T.append(1./(1. + math.exp(-value)))  # Don't allow values smaller than 1e-
        elif rng   == 'half':  # Range: 0 - 0.5
            if value < -16.:
                value = -16.
            values_T.append(0.5/(1. + math.exp(-value)))  # Don't allow values smaller than 1e-7
        elif rng == 'pos':  # Range: 0 - inf
            if value > 16.:
                value = 16.
            values_T.append(math.exp(value))  # Don't allow values bigger than ~ 1e7.
        elif rng == 'unc': # Range: - inf - inf.
            values_T.append(value)
    return np.array(values_T)

def trans_CU(values_T, param_ranges):
    'Transform parameters from constrained to unconstrained space.'
    if param_ranges[0] == 'all_unc':
        return values_T
    values_U = []
    for value, rng in zip(values_T, param_ranges):
        if rng   == 'unit':
            values_U.append(-math.log((1./value)-1))
        elif rng   == 'half':
            values_U.append(-math.log((0.5/value)-1))
        elif rng == 'pos':
            values_U.append(math.log(value))
        elif rng == 'unc':
            values_U.append(value)
    return np.array(values_U)

def grad_check(session, agent):
    'Check analytical likelihood gradient returned by logistic regression agent.'
    params = np.random.normal(0, 3., agent.n_params)
    lik_func  = lambda params: agent.session_likelihood(session, params, eval_grad = True)[0]
    grad_func = lambda params: agent.session_likelihood(session, params, eval_grad = True)[1]
    l2error = op.check_grad(lik_func, grad_func, params)
    print('Error between finite difference and analytic derivatives = ' + str(l2error))

# -------------------------------------------------------------------------------------
# Maximum likelihood fitting.
# -------------------------------------------------------------------------------------

def fit_session(session, agent, repeats = 5, brute_init = True, verbose = False):
    '''Find maximum likelihood parameter estimates for a session or list of sessions. '''

    if isinstance(agent, config_log_reg):  
        # Logistic regression models evaluate gradient and do not require parameter transformation.
        method = 'BFGS'
        calculates_gradient = True
        fit_func   = lambda params: agent.session_likelihood(session, params, sign = -1)
    else:
        # RL models do not calculate gradient and require parameter transformation from 
        # unconstrained to true space.
        method = 'Nelder-Mead'
        calculates_gradient = False
        fit_func   = lambda params: -agent.session_likelihood(session, trans_UC(params, agent.param_ranges))

    fits = []
    for i in range(repeats): # Perform fitting. 

        if agent.n_params <= 2 and i == 0 and brute_init: 
           # Initialise minimisation with brute force search.
           ranges = tuple([(-5,5) for i in range(agent.n_params)])
           init_params = op.brute(fit_func, ranges, Ns =  20, 
                                  full_output = True, finish = None)[0]
        else:
            init_params = np.random.normal(0, 3., agent.n_params)

        fits.append(op.minimize(fit_func, init_params, jac = calculates_gradient,
                                method = method, options = {'disp': verbose}))           

    fit = fits[np.argmin([f['fun'] for f in fits])]  # Select best fit out of repeats.

    session_fit = {'likelihood' : - fit['fun'],
                   'param_names': agent.param_names} 

    if isinstance(agent, config_log_reg):  
        session_fit['params'] = fit['x']
    else: # Transform parameters back to constrained space.
        session_fit['params'] = trans_UC(fit['x'], agent.param_ranges)

    return session_fit


def likelihood_landscape(session, agent, Ns = 5):
    'Evaluate likelihood landscape with brute force search.'
    fit_func   = lambda params: -agent.session_likelihood(session, trans_UC(params, agent.param_ranges))
    ranges = tuple([(-5,5) for i in range(agent.n_params)])
    lik_landscape = unpack_brute(op.brute(fit_func, ranges, Ns =  Ns, 
                                   full_output = True, finish = None), agent)
    return lik_landscape


def fit_sessions(sessions, agent):
    '''Perform maximum likelihood fitting on a list of sessions and return
    dictionary with fit information.  For logistic regression agents a one 
    sample ttest is used to check if each parameter loading is significantly
    different from zero.'''
    fit_list = [fit_session(session, agent) for session in sessions]
    fits = {'param_names': agent.param_names,
            'n_params'   : agent.n_params,
            'params'     : np.array([f['params'] for f in fit_list])}
    if isinstance(agent, config_log_reg):  
        fits['p_values'] = ttest_1samp(fits['params'],0)[1]
    return fits


def match_agent_parameters(agents, task, target = 'MB', target_params = [0.5,5.],
                           n_ses = 10, n_trials = 10000, use_median = True):
    '''Find parameters for agents which maximise likelihood of data simulated from 
    target agent with target parameters.'''
    from .simulation import simulated_sessions
    agents[target].params = target_params
    target_sessions = simulated_sessions(task, agents[target], n_ses, n_trials, target_params)
    matched_params = {target: target_params}
    for a_type in list(set(agents.keys()) - set([target])): # All agents except model based and Daw task agents.
        if agents[a_type].n_params > 0 and hasattr(agents[a_type], 'session_likelihood'):
            print('Matching parameters for agent: ' + a_type)
            session_params = [fit_session(s, agents[a_type])['params'] for s in target_sessions]
            if use_median:
                ave_params = np.median(session_params,0).tolist()
            else:
                ave_params = np.mean(session_params,0).tolist()
            agents[a_type].params  = ave_params
            matched_params[a_type] = ave_params
    return matched_params


# -------------------------------------------------------------------------------------
# Maximising performance (fraction rewarded trials) - Powell minimisation method
# -------------------------------------------------------------------------------------

def optimise_params(agent, task, n_sessions = 10, n_trials = 10000, repeats = 10, verbose = False, ftol = 0.001):
    '''Optimise agent parameters to maximise fraction of trials rewarded. Optimisation 
    is performed using Powell minimization with fixed random seed across all simulations in a
    given minimization run to give a smoother performance surface.  Once the optimal parameters for 
    this random seed are found, the performance is evaluated with these parameters but a 
    different random seed to avoid overestimating performance.
    '''
    
    from .simulation import simulated_session

    def negative_fraction_rewarded(params_U):
        np.random.seed(rand_seed);random.seed(rand_seed);
        agent.params = trans_UC(params_U, agent.param_ranges)
        return -simulated_session(task,agent,n_trials).fraction_rewarded

    print('Optimising params for agent ' + agent.name, end = '')
    session_fractions_rewarded, session_params = ([],[])
    rand_seed = 0 # Fixed random seed for all simulations in single optimize run.
    for s in range(n_sessions):
        print('.', end = '')
        if agent.n_params > 0:
            fits = []
            for i in range(repeats): # Perform fitting. 
                np.random.seed() # Seed from system time to get random intial parameters.
                init_params = np.random.normal(0, 3., agent.n_params)
                fits.append(op.minimize(negative_fraction_rewarded, init_params, method = 'Powell',
                            options = {'disp': verbose, 'xtol': 0.01, 'ftol':ftol}))           
            best_fit = fits[np.argmin([f['fun'] for f in fits])]  # Select best performance out of repeats.            
            session_params.append(trans_UC(best_fit['x'],agent.param_ranges))
            rand_seed += 1 
            session_fractions_rewarded.append(
                simulated_session(task,agent,n_trials, 
                                  params = session_params[-1]).fraction_rewarded)
        else:
            session_fractions_rewarded.append(simulated_session(task,agent,n_trials).fraction_rewarded)
    print('')
    return {'mean':np.mean(session_fractions_rewarded),
            'SEM' :sem(session_fractions_rewarded),
            'session_fractions_rewarded':session_fractions_rewarded,
            'session_params': session_params,
            'mean_params': np.mean(session_params,0)}

def optimise_agents_parameters(agents, task, n_trials = 10000, n_sessions = 10, verbose = False,
                             agents_to_compare = ['Q(1)', 'Q(0)', 'MB', 'RF', 'LS', 'RND']):
    agents_to_compare = list(set(agents_to_compare).intersection(agents.keys()))
    performance = {}
    for a_type in agents_to_compare:  # Evaluate performance.
        performance[a_type] = optimise_params(agents[a_type], task, n_sessions, n_trials, verbose = verbose)
    performance = _performance_p_values(performance, agents_to_compare)
    return performance 

def _performance_p_values(performance, agents_to_compare):
    'Significance testing of performance differences.'
    means = [performance[a_type]['mean'] for a_type in agents_to_compare]
    agents_sorted = [agents_to_compare[i] for i in np.argsort(means)]
    p_values = np.ones([len(agents_sorted), len(agents_sorted)])
    for i, a_type_i in enumerate(agents_sorted):  # Evaluate p values for performance differences.
        for j, a_type_j in enumerate(agents_sorted):
            if i != j:
                p_values[i,j] = ttest_ind(performance[a_type_i]['session_fractions_rewarded'],
                                          performance[a_type_j]['session_fractions_rewarded'])[1]
    performance['agents_sorted'] = agents_sorted
    performance['p_values'] = p_values       
    return performance

# -------------------------------------------------------------------------------------
# Maximising performance (fraction rewarded trials) - Grid search method
# -------------------------------------------------------------------------------------


def optimise_params_gs(agent, task, n_sessions = 10, n_trials = 10000, Ns = 20, return_search = False):
    'Optimise parameters for agent using a brute force grid search.'
    from .simulation import simulated_sessions

    def eval_performance(params_U = [], return_full = False):
        'Returns the negative fraction correct.'
        if len(params_U) > 0:
            agent.params = trans_UC(params_U, agent.param_ranges)
        sessions = simulated_sessions(task, agent, n_sessions, n_trials, verbose = False)
        mean_perf = -np.mean([s.fraction_rewarded for s in sessions]) 
        if return_full:
            sessions_perf = np.array([s.fraction_rewarded for s in sessions])
            SEM_perf = sem(sessions_perf)
            return(mean_perf, SEM_perf, sessions_perf)
        else:
            return mean_perf

    if agent.n_params > 0:
        ranges = tuple([(-8,8) for i in range(agent.n_params)])
        perf_search  = unpack_brute(op.brute(eval_performance, ranges, Ns =  Ns, 
                                       full_output = True, finish = None), agent)
        if return_search:
            return perf_search
        else:
            params = perf_search['max_params_U']
            opt_perf, opt_SEM, session_frac_rew =  eval_performance(params, True)
    else:
        params = None
        opt_perf, opt_SEM, session_frac_rew =  eval_performance([], True)
    return -opt_perf, opt_SEM, session_frac_rew, params


def optimise_agents_parameters_gs(agents, task, n_trials = 10000, n_sessions = 10, grid_N = 20,
                              agents_to_compare = ['RND', 'Q(1)', 'Q(0)', 'MB']):
    'Find parameters which maximize fraction of rewarded trials for each agent.'
    agents_to_compare = list(set(agents_to_compare).intersection(agents.keys()))
    performance = {}
    for a_type in agents_to_compare:  # Evaluate performance.
        print('Optimising parameters for agent: ' + a_type)
        opt_perf, opt_SEM, session_frac_rew, params = optimise_params_gs(agents[a_type], task, n_sessions = n_sessions,
                                                 n_trials = n_trials, Ns = grid_N)
        performance[a_type]= {'mean': opt_perf, 'SEM': opt_SEM,
                              'session_fractions_rewarded': session_frac_rew, 'params': params}
    performance = _performance_p_values(performance, agents_to_compare)
    return performance

def unpack_brute(opt, agent):# Unpack output of brute force search.
    return    {'max_params_U': opt[0],      
               'max_params'  : trans_UC(opt[0],agent.param_ranges),
               'fun_max'     : -opt[1],
               'X_grid_U'    : opt[2][0,:,:],
               'Y_grid_U'    : opt[2][1,:,:],
               'fun'         : -opt[3],
               'param_names' : agent.param_names}



