import numpy as np
from .plotting import session_plot, session_stay_probs
from . import model_fitting as mf
from .logistic_regression import COTI_model, CCOTI_model
from .tasks import Orig_two_step

# -------------------------------------------------------------------------------------
# Simulated data generation.
# -------------------------------------------------------------------------------------

class simulated_session():
    '''Stores agent parameters and simulated data.
    '''
    def __init__(self, task, agent, n_trials = 1000, params = None):
        '''Simulate session with current agent and task parameters.'''
        self.n_trials = n_trials
        self.name = agent.name
        if params is not None:
            agent.params = params
        if isinstance(task, Orig_two_step): # Original version of task with seconds step choice.
            choices, second_steps, choices_s, outcomes = agent.simulate(task, n_trials)
        else:
            choices, second_steps, outcomes = agent.simulate(task, n_trials)
            choices_s = None
        self.CTSO = {'choices'      : choices,
                     'transitions'  : (choices == second_steps).astype(int),
                     'second_steps' : second_steps,
                     'choices_s'   : choices_s,
                     'outcomes'     : outcomes}
        self.reward_probs = task.reward_probs
        self.stay_probs = session_stay_probs(self)
        self.fraction_rewarded = np.mean(outcomes)

    def plot(self, fig_no = 1):session_plot(self, fig_no)

    def CTSO_unpack(self, ret_type = int):
        'Unpack CTSO dict to seperate numpy arrays of specified type.'
        if ret_type == int:
            return (self.CTSO['choices']     , self.CTSO['transitions'], 
                    self.CTSO['second_steps'], self.CTSO['outcomes'])
        else:
            return (self.CTSO['choices'].astype(ret_type)     , self.CTSO['transitions'].astype(ret_type),
                    self.CTSO['second_steps'].astype(ret_type), self.CTSO['outcomes'].astype(ret_type))

        
def simulated_sessions(task, agent, n_ses = 10, n_trials = 1000, params = None, verbose = True):
    'Return a list of n_ses simulated sessions each of length n_trials.'
    if verbose: print('Simulating: ' + agent.name)
    if params:agent.params = params
    return [simulated_session(task, agent, n_trials) for s in range(n_ses)]


def param_grid_sim(task, agent, n_ses = 10, n_trials = 1000, grid_N = 5, reg_strength = 0.01):
    '''Simulate agent on task over grid of parameter values and evaluate fit of 
    COTI logistic regression agent for each simulation.  Regularisation for 
    logistic regression fits can be specified with reg_strength parameter.'''
    assert(len(agent.params) == 2), 'Grid simulation only valid for agents with two parameters.'
    if reg_strength: # Turn on regularisation
        COTI_model.reg_strength = reg_strength
    param_0_values = _get_param_values(agent.param_ranges[0], grid_N)
    param_1_values = _get_param_values(agent.param_ranges[1], grid_N) 
    grid_sim = np.zeros([grid_N, grid_N], dtype = object)
    print('Grid simulation: ' + agent.name, end = '')
    for i, p0 in enumerate(param_0_values):
        for j, p1, in enumerate(param_1_values):
            print('.', end = '')
            sessions = simulated_sessions(task, agent, n_ses, n_trials, params = [p0, p1], verbose = False)            
            COTI_fit  = mf.fit_sessions(sessions,  COTI_model)
            CCOTI_fit = mf.fit_sessions(sessions, CCOTI_model)
            grid_sim[i,j] = {'sessions'   : sessions,
                             'COTI_fit'   : COTI_fit,
                             'CCOTI_fit'  : CCOTI_fit,
                             'params'     : [p0, p1]}
    print('')
    grid_sim[0,0]['param_names'] = agent.param_names
    grid_sim[0,0]['param_grid'] = [param_0_values, param_1_values]
    if reg_strength: # Turn off regularisation
        COTI_model.reg_strength = 0
    return grid_sim


def _get_param_values(param_range, grid_N):
    if param_range == 'unit':
        return np.linspace(0, 1 , grid_N + 2)[1:-1]
    elif param_range == 'pos':
        return np.linspace(0, 10, grid_N + 2)[1:-1]
    elif param_range == 'half':
        return np.linspace(0, 0.5, grid_N + 2)[1:-1]




