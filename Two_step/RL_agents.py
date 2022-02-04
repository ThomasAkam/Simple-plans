import numpy as np
from numba import jit
from random import random, randint
import sys

log_max_float = np.log(sys.float_info.max/2.1) # Log of largest possible floating point number.


# Note: Numba is used to speed up likelihood evaluation. If numba is not available, 
# remove @jit decorators and the code will run, but with much slower likelihood evaluation.

# -------------------------------------------------------------------------------------
# Utility functions.
# -------------------------------------------------------------------------------------

def softmax(Q,T):
    "Softmax choice probs given values Q and inverse temp T."
    QT = Q * T
    QT[QT > log_max_float] = log_max_float # Protection against overflow in exponential.    
    expQT = np.exp(QT)
    return expQT/expQT.sum()


def array_softmax(Q,T):
    '''Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([n_trials,2])
    T: Inverse temp  - float.'''
    P = np.zeros(Q.shape)
    TdQ = -T*(Q[:,0]-Q[:,1])
    TdQ[TdQ > log_max_float] = log_max_float # Protection against overflow in exponential.    
    P[:,0] = 1./(1. + np.exp(TdQ))
    P[:,1] = 1. - P[:,0]
    return P

def protected_log(x):
    'Return log of x protected against giving -inf for very small values of x.'
    return np.log(((1e-200)/2)+(1-(1e-200))*x)

def choose(P):
    "Takes vector of probabilities P summing to 1, returns integer s with prob P[s]"
    return sum(np.cumsum(P) < random())

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#
# Agents for reduced task version without choice at second step.
#
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Q1 agent
# -------------------------------------------------------------------------------------

class Q1():
    ''' A Q1 (direct reinforcement) agent, in which the outcome directly increaces
    or decreaces the value of the action chosen at the first step.
    '''

    def __init__(self):

        self.name = 'Q1'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td = np.zeros(2)
        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_td, iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_td[c] = (1. - alpha) * Q_td[c] +  alpha * o    

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params, return_Qs = False):

        # Unpack trial events.
        choices, outcomes = (session.CTSO['choices'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.
        Q_td = np.zeros([session.n_trials + 1, 2])  # Model free action values.

        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.

            nc = 1 - c # Not chosen action.

            # update action values.
            Q_td[i+1, c] = (1. - alpha) * Q_td[i,c] +  alpha * o   
            Q_td[i+1,nc] = Q_td[i,nc]
            
        # Evaluate choice probabilities and likelihood. 
        choice_probs = array_softmax(Q_td, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)
        if return_Qs:
            return Q_td
        else:
            return session_log_likelihood

# -------------------------------------------------------------------------------------
# Q1 perseveration agent.
# -------------------------------------------------------------------------------------

class Q1_prsv():
    ''' A Q1 agent with a perseveration bias.'''

    def __init__(self, params = None):

        self.name = 'Q1_prsv'
        self.param_names  = ['alpha', 'iTemp', 'prsv']
        self.params       = [ 0.5   ,  5.    ,  0.2 ]  
        self.param_ranges = ['unit' , 'pos'  , 'unc' ]
        self.n_params = 2
        if params:
            self.params = params

    def simulate(self, task, n_trials):

        alpha, iTemp, prsv = self.params  

        Q_td  = np.zeros(2) # TD action values excluding perseveration bias.
        Q_net = np.zeros(2) # Net action value including perseveration bias.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_net, iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_td[c] = (1. - alpha) * Q_td[c] +  alpha * o  
            Q_net[:] = Q_td[:]
            Q_net[c] += 1. * prsv

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

# -------------------------------------------------------------------------------------
# Q0 agent
# -------------------------------------------------------------------------------------

class Q0():
    ''' A temporal difference agent without elegibility traces.'''

    def __init__(self):

        self.name = 'Q0'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_f = np.zeros(2) # First  step action values. 
        Q_td_s = np.zeros(2) # Second step action values.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_td_f, iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * Q_td_s[s]   
            Q_td_s[s] = (1. - alpha) * Q_td_s[s] +  alpha * o     

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.
        Q_td_f = np.zeros([session.n_trials + 1, 2])  # First  step action values.
        Q_td_s = np.zeros([session.n_trials + 1, 2])  # Second step action values.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            nc = 1 - c # Not chosen first step action.
            ns = 1 - s # Not reached second step state.

            # update action values.
            Q_td_f[i+1,c] = (1. - alpha) * Q_td_f[i,c] +  alpha * Q_td_s[i,s]  
            Q_td_s[i+1,s] = (1. - alpha) * Q_td_s[i,s] +  alpha * o    
            Q_td_f[i+1,nc] = Q_td_f[i,nc]
            Q_td_s[i+1,ns] = Q_td_s[i,ns]
            
        # Evaluate choice probabilities and likelihood. 
        choice_probs = array_softmax(Q_td_f, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        return session_log_likelihood


# -------------------------------------------------------------------------------------
# TD lambda agent
# -------------------------------------------------------------------------------------

class Q_lambda():
    ''' A temporal difference agent with adjustable elegibility trace.
    '''

    def __init__(self, lambd = 0.5):

        self.name = 'Q_lambda'
        self.lambd = lambd
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_f = np.zeros(2) # First  step action values. 
        Q_td_s = np.zeros(2) # Second step action values.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_td_f, iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * (Q_td_s[s] + self.lambd * (o - Q_td_s[s]))  
            Q_td_s[s] = (1. - alpha) * Q_td_s[s] +  alpha * o     

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  
        lambd = self.lambd

        #Variables.
        Q_td_f = np.zeros([session.n_trials + 1, 2])  # First  step action values.
        Q_td_s = np.zeros([session.n_trials + 1, 2])  # Second step action values.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            nc = 1 - c # Not chosen first step action.
            ns = 1 - s # Not reached second step state.

            # update action values.
            Q_td_f[i+1,c] = (1. - alpha) * Q_td_f[i,c] +  alpha * (Q_td_s[i,s] + lambd * (o - Q_td_s[i,s]))
            Q_td_s[i+1,s] = (1. - alpha) * Q_td_s[i,s] +  alpha * o    
            Q_td_f[i+1,nc] = Q_td_f[i,nc]
            Q_td_s[i+1,ns] = Q_td_s[i,ns]
            
        # Evaluate choice probabilities and likelihood. 
        choice_probs = array_softmax(Q_td_f, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        return session_log_likelihood



# -------------------------------------------------------------------------------------
# Model based agent.
# -------------------------------------------------------------------------------------

class Model_based():
    ''' A model based agent which learns the values of the second step states 
    through reward prediction errors and then calculates the values of the first step
    actions using the transition probabilties.
    '''

    def __init__(self):
        self.name = 'Model based'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_s  = np.zeros(2)  # Second_step state values.
        Q_mb = np.zeros(2)  # Model based action values.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mb, iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_s[s] = (1. - alpha) * Q_s[s] +  alpha * o
            Q_mb   = 0.8 * Q_s + 0.2 * Q_s[::-1]

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.
        Q_s    = np.zeros([session.n_trials + 1, 2])  # Second_step state values.
        Q_mb   = np.zeros([session.n_trials + 1, 2])  # Model based action values.


        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            ns = 1 - s # Not reached second step state.

            # update action values.
            Q_s[i+1, s] = (1. - alpha) * Q_s[i,s] +  alpha * o
            Q_s[i+1,ns] = Q_s[i,ns]
            
        # Evaluate choice probabilities and likelihood. 

        Q_mb = 0.8 * Q_s + 0.2 * Q_s[:,::-1]
        choice_probs = array_softmax(Q_mb, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        return session_log_likelihood

# -------------------------------------------------------------------------------------
# Model based perseveration agent.
# -------------------------------------------------------------------------------------

class MB_prsv():
    ''' A model based agent with perseveration bias'''

    def __init__(self, params = None):
        self.name = 'MB_prsv'
        self.param_names  = ['alpha', 'iTemp', 'prsv']
        self.params       = [ 0.5   ,  5.    ,  0.2  ]  
        self.param_ranges = ['unit' , 'pos'  , 'unc' ]
        self.n_params = 3
        if params:
            self.params = params

    def simulate(self, task, n_trials):

        alpha, iTemp, prsv = self.params  

        Q_s   = np.zeros(2)  # Second_step state values.
        Q_mb  = np.zeros(2)  # Model based action values.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mb, iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_s[s] = (1. - alpha) * Q_s[s] +  alpha * o
            Q_mb   = 0.8 * Q_s + 0.2 * Q_s[::-1]
            Q_mb[c] += 1. * prsv # Perseveration bias.

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

# -------------------------------------------------------------------------------------
# Model based agent with transition matrix leraning.
# -------------------------------------------------------------------------------------

class MB_trans_learn():
    ''' Model based agent which learns the transtion matrix from experience.
    '''

    def __init__(self, params = None):
        self.name = 'MB trans. learn.'
        self.param_names  = ['alpha', 'iTemp', 'tlr' ]
        self.params       = [ 0.5   ,  5.    ,  0.5  ]  
        self.param_ranges = ['unit' , 'pos'  , 'unit']
        self.n_params = 3
        if params:
            self.params = params

    def simulate(self, task, n_trials):

        alpha, iTemp, tlr = self.params  

        Q_s  = np.zeros(2)      # Second_step state values.
        Q_mb = np.zeros(2)      # Model based action values.
        tp   = np.ones(2) * 0.5 # Transition probabilities for first step actions, coded as 
                                # probability of reaching second step state 0.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mb, iTemp)) 
            s, o = task.trial(c)
            # update action values and transition probabilities.
            Q_s[s] = (1. - alpha) * Q_s[s] +  alpha * o
            tp[c]  = (1. - tlr) * tp[c] + tlr * (s == 0)
            Q_mb   = tp * Q_s[0] + (1 - tp) * Q_s[1]

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

# -------------------------------------------------------------------------------------
# Reward_as_cue agent
# -------------------------------------------------------------------------------------

class Reward_as_cue():
    '''Agent which uses reward location as a cue for state.  The agent learns seperate values
    for actions following distinct outcomes and second_steps on the previous trial.
    '''

    def __init__(self):

        self.name = 'Reward as cue'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.05   ,  6.   ] 
        self.param_ranges = ['unit' , 'pos'  ] 
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td = np.zeros([2, 2, 2]) # Indicies: action, prev. second step., prev outcome
        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        ps = randint(0,1) # previous second step.
        po = randint(0,1) # previous outcome.

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_td[:,ps,po], iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_td[c,ps,po] = (1. - alpha) * Q_td[c,ps,po] +  alpha * o  
            ps, po = s, o  

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.
        Q_td = np.zeros([2, 2, 2]) # Indicies: action, prev. second step., prev outcome
        Q    = np.zeros([session.n_trials, 2])  # Active action values.

        ps = randint(0,1) # previous second step.
        po = randint(0,1) # previous outcome.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            Q[i,0] = Q_td[0,ps,po]
            Q[i,1] = Q_td[1,ps,po]

            # update action values.
            Q_td[c,ps,po] = (1. - alpha) * Q_td[c,ps,po] +  alpha * o  
            ps, po = s, o  

        # Evaluate choice probabilities and likelihood. 

        choice_probs = array_softmax(Q, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        return session_log_likelihood

# -------------------------------------------------------------------------------------
# Reward_as_cue_fixed agent
# -------------------------------------------------------------------------------------

class RAC_fixed():
    '''Agent which deterministically follows the following decison rules
    mapping second step and outcome onto next choice (s, o --> c)  .   
    0, 1 --> 0
    0, 0 --> 1
    1, 1 --> 1
    1, 0 --> 0
    '''

    def __init__(self):

        self.name = 'RAC fixed.'
        self.param_names  = []
        self.params       = []  
        self.n_params = 0

    def simulate(self, task, n_trials):

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        ps = randint(0,1) # previous second step.
        po = randint(0,1) # previous outcome.

        for i in range(n_trials):
            # Generate trial events.
            c = int(ps == po)
            s, o = task.trial(c)
            ps, po = s, o  

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

# -------------------------------------------------------------------------------------
# Latent state agent
# -------------------------------------------------------------------------------------

class Latent_state():
    
    '''Agent which belives that there are two states of the world:

    State 0, Second step 0 reward prob = good_prob, sec. step 1 reward prob = 1 - good_prob
    State 1, Second step 1 reward prob = good_prob, sec. step 0 reward prob = 1 - good_prob

    Agent believes the probability that the state of the world changes on each step is p_r.

    The agent infers which state of the world it is most likely to be in, and then chooses 
    the action which leads to the best second step in that state with probability (1- p_lapse)
    '''

    def __init__(self):

        self.name = 'Latent state'
        self.param_names  = ['p_r' , 'p_lapse']
        self.params       = [ 0.1  , 0.1      ]
        self.param_ranges = ['half', 'half'   ]   
        self.n_params = 2

    def simulate(self, task, n_trials):

        p_r, p_lapse = self.params
        good_prob = 0.8

        p_1 = 0.5 # Probability world is in state 1.

        p_o_1 = np.array([[good_prob    , 1 - good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - good_prob, good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1               # Probability of observed outcome given world in state 0.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):

            # Generate trial events.
            c = (p_1 > 0.5) == (random() > p_lapse)
            s, o = task.trial(c)
            # Bayesian update of state probabilties given observed outcome.
            p_1 = p_o_1[s,o] * p_1 / (p_o_1[s,o] * p_1 + p_o_0[s,o] * (1 - p_1))   
            # Update of state probabilities due to possibility of block reversal.
            p_1 = (1 - p_r) * p_1 + p_r * (1 - p_1)  


            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes


    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        p_r, p_lapse = params
        good_prob = 0.8

        p_o_1 = np.array([[good_prob    , 1 - good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - good_prob, good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1               # Probability of observed outcome given world in state 0.

        p_1    = np.zeros(session.n_trials + 1) # Probability world is in state 1.
        p_1[0] = 0.5 

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            # Bayesian update of state probabilties given observed outcome.
            p_1[i+1] = p_o_1[s,o] * p_1[i] / (p_o_1[s,o] * p_1[i] + p_o_0[s,o] * (1 - p_1[i]))   
            # Update of state probabilities due to possibility of block reversal.
            p_1[i+1] = (1 - p_r) * p_1[i+1] + p_r * (1 - p_1[i+1])  

        # Evaluate choice probabilities and likelihood. 
        choice_probs = np.zeros([session.n_trials + 1, 2])
        choice_probs[:,1] = (p_1 > 0.5) * (1 - p_lapse) + (p_1 <= 0.5) * p_lapse
        choice_probs[:,0] = 1 - choice_probs[:,1] 
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        return session_log_likelihood

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#
# Agents for original (Daw et al 2011) task version.
#
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Q1 agent - original task version.
# -------------------------------------------------------------------------------------

class Q1_orig():
    ''' A direct reinforcement (Q1) agent for use with the version of the task used in 
    Daw et al 2011 with choices at the second step.
    '''

    def __init__(self):

        self.name = 'Q1, original task.'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_f = np.zeros(2)     # First step action values.
        Q_td_s = np.zeros([2,2]) # Second step action values, indicies: [state, action]

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c   = choose(softmax(Q_td_f, iTemp))      # First step action.
            s   = task.first_step(c)                  # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp)) # Second_step action.
            o   = task.second_step(s, c_s)            # Trial outcome.   

            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * o    
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, choices_s, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['choices_s'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.

        Q_td_f = np.zeros(2)     # First step action values.
        Q_td_s = np.zeros([2,2]) # Second step action values, indicies: [state, action]

        Q_td_f_array = np.zeros([session.n_trials + 1, 2])     # First step action values.
        Q_td_s_array = np.zeros([session.n_trials + 1, 2])     # Second step action values for second step reached on each trial.


        for i, (c, s, c_s, o) in enumerate(zip(choices, second_steps, choices_s, outcomes)): # loop over trials.

            # Store action values in array, note: inner loop over actions is used rather than 
            # slice indexing because slice indexing is not currently supported by numba.

            for j in range(2):
                Q_td_f_array[i,j] = Q_td_f[j]
                Q_td_s_array[i,j] = Q_td_s[s,j]

            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * o    
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

        # Evaluate choice probabilities and likelihood. 
        choice_probs_f = array_softmax(Q_td_f_array, iTemp)
        choice_probs_s = array_softmax(Q_td_s_array, iTemp)
        trial_log_likelihood_f = protected_log(choice_probs_f[np.arange(session.n_trials), choices  ])
        trial_log_likelihood_s = protected_log(choice_probs_s[np.arange(session.n_trials), choices_s])
        session_log_likelihood = np.sum(trial_log_likelihood_f) + np.sum(trial_log_likelihood_s)  
        return session_log_likelihood

# -------------------------------------------------------------------------------------
# Q0 agent - original task version.
# -------------------------------------------------------------------------------------

class Q0_orig():
    ''' Q(0) agent for use with the version of the task used in Daw et al 2011.'''

    def __init__(self):

        self.name = 'Q0, original task.'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_f = np.zeros(2)     # First step action values.
        Q_td_s = np.zeros([2,2]) # Second step action values, indicies: [state, action]

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c   = choose(softmax(Q_td_f, iTemp))      # First step action.
            s   = task.first_step(c)                  # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp)) # Second_step action.
            o   = task.second_step(s, c_s)            # Trial outcome.   

            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * np.max(Q_td_s[s,:])  
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, choices_s, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['choices_s'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.

        Q_td_f = np.zeros(2)     # First step action values.
        Q_td_s = np.zeros([2,2]) # Second step action values, indicies: [state, action]

        Q_td_f_array = np.zeros([session.n_trials + 1, 2])     # First step action values.
        Q_td_s_array = np.zeros([session.n_trials + 1, 2])     # Second step action values for second step reached on each trial.


        for i, (c, s, c_s, o) in enumerate(zip(choices, second_steps, choices_s, outcomes)): # loop over trials.

            # Store action values in array, note: inner loop over actions is used rather than 
            # slice indexing because slice indexing is not currently supported by numba.

            for j in range(2):
                Q_td_f_array[i,j] = Q_td_f[j]
                Q_td_s_array[i,j] = Q_td_s[s,j]

            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * np.max(Q_td_s[s,:])  
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

        # Evaluate choice probabilities and likelihood. 
        choice_probs_f = array_softmax(Q_td_f_array, iTemp)
        choice_probs_s = array_softmax(Q_td_s_array, iTemp)
        trial_log_likelihood_f = protected_log(choice_probs_f[np.arange(session.n_trials), choices  ])
        trial_log_likelihood_s = protected_log(choice_probs_s[np.arange(session.n_trials), choices_s])
        session_log_likelihood = np.sum(trial_log_likelihood_f) + np.sum(trial_log_likelihood_s)  
        return session_log_likelihood

# -------------------------------------------------------------------------------------
# Model based agent - original task version.
# -------------------------------------------------------------------------------------

class Model_based_orig():
    ''' A model based agent for use with the version of the task used in 
    Daw et al 2011 with choices at the second step.
    '''

    def __init__(self):
        self.name = 'Model based, original task.'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_s = np.zeros([2,2])  # Second_step action values, indicies: [state, action]
        Q_mb   = np.zeros(2)      # Model based action values.

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mb, iTemp)) 
            s   = task.first_step(c)                  # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp)) # Second_step action.
            o   = task.second_step(s, c_s)            # Trial outcome.  

            # update action values.
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o
            Q_s = np.max(Q_td_s, 1) # State values are max action value available in each state.
            Q_mb   = 0.7 * Q_s + 0.3 * Q_s[::-1]

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, choices_s, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['choices_s'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.
        Q_mb = np.zeros(2)        # Model based action values.
        Q_td_s = np.zeros([2,2])  # Second_step action values, indicies: [state, action]

        Q_mb_array   = np.zeros([session.n_trials + 1, 2])     # Model based action values.
        Q_td_s_array = np.zeros([session.n_trials + 1, 2])     # Second step action values for second step reached on each trial.


        for i, (c, s, c_s, o) in enumerate(zip(choices, second_steps, choices_s, outcomes)): # loop over trials.

            # Store action values in array, note: inner loop over actions is used rather than 
            # slice indexing because slice indexing is not currently supported by numba.

            for j in range(2):
                Q_mb_array[i,j] = Q_mb[j]
                Q_td_s_array[i,j] = Q_td_s[s,j]

            # update action values.
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o
            Q_s_0 = np.max(Q_td_s[0,:]) # Max action value in second step state 0.
            Q_s_1 = np.max(Q_td_s[1,:]) # Max action value in second step state 1.
            Q_mb[0] = 0.7 * Q_s_0 + 0.3 * Q_s_1
            Q_mb[1] = 0.3 * Q_s_0 + 0.7 * Q_s_1  

        # Evaluate choice probabilities and likelihood. 
        choice_probs_f = array_softmax(Q_mb_array, iTemp)
        choice_probs_s = array_softmax(Q_td_s_array, iTemp)
        trial_log_likelihood_f = protected_log(choice_probs_f[np.arange(session.n_trials), choices  ])
        trial_log_likelihood_s = protected_log(choice_probs_s[np.arange(session.n_trials), choices_s])
        session_log_likelihood = np.sum(trial_log_likelihood_f) + np.sum(trial_log_likelihood_s)  
        return session_log_likelihood

# -------------------------------------------------------------------------------------
# Reward_as_cue agent - original task version.
# -------------------------------------------------------------------------------------

class Reward_as_cue_orig():
    '''Reward as cue agent for original version of task.
    '''

    def __init__(self):

        self.name = 'Reward as cue, original task.'
        self.param_names  = ['alpha', 'iTemp', 'alpha_s', 'iTemp_s']
        self.params       = [ 0.05  ,   10.  ,  0.5     ,     5.   ] 
        self.param_ranges = ['unit' , 'pos'  , 'unit'   ,   'pos'  ] 
        self.n_params = 4

    def simulate(self, task, n_trials):

        alpha, iTemp, alpha_s, iTemp_s = self.params  

        Q_td_f = np.zeros([2, 2, 2]) # First step action values, indicies: action, prev. second step., prev outcome
        Q_td_s = np.zeros([2,2])     # Second step action values, indicies: [state, action]

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        ps = randint(0,1) # previous second step.
        po = randint(0,1) # previous outcome.

        for i in range(n_trials):
            # Generate trial events.
            c   = choose(softmax(Q_td_f[:,ps,po], iTemp)) # First step action.
            s   = task.first_step(c)                    # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp_s))   # Second step action.
            o   = task.second_step(s, c_s)              # Trial outcome.   

            # update action values.
            Q_td_f[c,ps,po] = (1. - alpha) * Q_td_f[c,ps,po] +  alpha * o  
            ps, po = s, o  
            Q_td_s[s,c_s] = (1. - alpha_s) * Q_td_s[s,c_s] + alpha_s * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, choices_s, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['choices_s'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp, alpha_s, iTemp_s = params  

        #Variables.
        Q_td_f = np.zeros([2, 2, 2]) # First step action values, indicies: action, prev. second step., prev outcome
        Q_td_s = np.zeros([2,2])     # Second step action values, indicies: [state, action]

        Q_td_f_array = np.zeros([session.n_trials + 1, 2])     # First step action values.
        Q_td_s_array = np.zeros([session.n_trials + 1, 2])     # Second step action values for second step reached on each trial.

        ps = 0 # previous second step.
        po = 0 # previous outcome.

        for i, (c, s, c_s, o) in enumerate(zip(choices, second_steps, choices_s, outcomes)): # loop over trials.

            # Store action values in array, note: inner loop over actions is used rather than 
            # slice indexing because slice indexing is not currently supported by numba.

            for j in range(2):
                Q_td_f_array[i,j] = Q_td_f[j, ps, po]
                Q_td_s_array[i,j] = Q_td_s[s,j]

            # update action values.
            Q_td_f[c,ps,po] = (1. - alpha) * Q_td_f[c,ps,po] +  alpha * o  
            ps, po = s, o  
            Q_td_s[s,c_s] = (1. - alpha_s) * Q_td_s[s,c_s] + alpha_s * o

        # Evaluate choice probabilities and likelihood. 
        choice_probs_f = array_softmax(Q_td_f_array, iTemp)
        choice_probs_s = array_softmax(Q_td_s_array, iTemp_s)
        trial_log_likelihood_f = protected_log(choice_probs_f[np.arange(session.n_trials), choices  ])
        trial_log_likelihood_s = protected_log(choice_probs_s[np.arange(session.n_trials), choices_s])
        session_log_likelihood = np.sum(trial_log_likelihood_f) + np.sum(trial_log_likelihood_s)  
        return session_log_likelihood


# -------------------------------------------------------------------------------------
# Reward_as_cue_fixed agent
# -------------------------------------------------------------------------------------

class RAC_fixed_orig():
    '''RAC_fixed agent for original task version, using deterministic strategy at first
    step then TD at second step.
    '''

    def __init__(self):

        self.name = 'RAC fixed, original task'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        Q_td_s = np.zeros([2,2])     # Second step action values, indicies: [state, action]

        task.reset(n_trials)

        ps = randint(0,1) # previous second step.
        po = randint(0,1) # previous outcome.

        for i in range(n_trials):
            # Generate trial events.
            c = int(ps == po)
            s   = task.first_step(c)                    # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp))   # Second step action.
            o   = task.second_step(s, c_s)              # Trial outcome.   
            ps, po = s, o  

            # update action values.
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes


# -------------------------------------------------------------------------------------
# Latent state agent - original task version.
# -------------------------------------------------------------------------------------

class Latent_state_orig():
    
    '''Version of latent state agent for original version of task.

    Agent belives that there are two states of the world:

    State 0, Second step 0 reward prob = good_prob, sec. step 1 reward prob = 1 - good_prob
    State 1, Second step 1 reward prob = good_prob, sec. step 0 reward prob = 1 - good_prob

    Agent believes the probability that the state of the world changes on each step is p_r.

    The agent infers which state of the world it is most likely to be in, and then chooses 
    the action which leads to the best second step in that state with probability (1- p_lapse)

    Choices at the second step are mediated by action values learnt through TD.
    '''

    def __init__(self):

        self.name = 'Latent state, original task.'
        self.param_names  = ['p_r' , 'p_lapse', 'alpha', 'iTemp']
        self.params       = [ 0.1  , 0.1      ,  0.5   ,    6.  ]
        self.param_ranges = ['half', 'half'   , 'unit' , 'pos'  ]   
        self.n_params = 4

    def simulate(self, task, n_trials):

        p_r, p_lapse, alpha, iTemp = self.params
        good_prob = 0.625

        p_1 = 0.5 # Probability world is in state 1.

        p_o_1 = np.array([[good_prob    , 1 - good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - good_prob, good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1               # Probability of observed outcome given world in state 0.

        Q_td_s = np.zeros([2,2])     # Second step action values, indicies: [state, action]

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):

            # Generate trial events.
            c   = (p_1 > 0.5) == (random() > p_lapse) # First step choice.
            s   = task.first_step(c)                  # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp)) # Second step action.
            o   = task.second_step(s, c_s)            # Trial outcome.   

            # Bayesian update of state probabilties given observed outcome.
            p_1 = p_o_1[s,o] * p_1 / (p_o_1[s,o] * p_1 + p_o_0[s,o] * (1 - p_1))   
            # Update of state probabilities due to possibility of block reversal.
            p_1 = (1 - p_r) * p_1 + p_r * (1 - p_1)  

            # Update second step action values.

            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes


    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, choices_s, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['choices_s'], session.CTSO['outcomes'])

        # Unpack parameters.
        p_r, p_lapse, alpha, iTemp = params
        good_prob = 0.625

        p_o_1 = np.array([[good_prob    , 1 - good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - good_prob, good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1               # Probability of observed outcome given world in state 0.

        p_1    = np.zeros(session.n_trials + 1) # Probability world is in state 1.
        p_1[0] = 0.5 

        Q_td_s = np.zeros([2,2])     # Second step action values, indicies: [state, action]
        Q_td_s_array = np.zeros([session.n_trials + 1, 2])     # Second step action values for second step reached on each trial.


        for i, (c, s, c_s, o) in enumerate(zip(choices, second_steps, choices_s, outcomes)): # loop over trials.

            # Bayesian update of state probabilties given observed outcome.
            p_1[i+1] = p_o_1[s,o] * p_1[i] / (p_o_1[s,o] * p_1[i] + p_o_0[s,o] * (1 - p_1[i]))   
            # Update of state probabilities due to possibility of block reversal.
            p_1[i+1] = (1 - p_r) * p_1[i+1] + p_r * (1 - p_1[i+1])  

            for j in range(2): # Store second step action values in array
                Q_td_s_array[i,j] = Q_td_s[s,j]

            # Update second step action values.

            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o


        # Evaluate choice probabilities and likelihood. 
        choice_probs = np.zeros([session.n_trials + 1, 2]) # First step choice probabilities.
        choice_probs[:,1] = (p_1 > 0.5) * (1 - p_lapse) + (p_1 <= 0.5) * p_lapse
        choice_probs[:,0] = 1 - choice_probs[:,1] 
        choice_probs_s = array_softmax(Q_td_s_array, iTemp) # Second step choice probabilities.
        trial_log_likelihood_f = protected_log(choice_probs[np.arange(session.n_trials), choices])
        trial_log_likelihood_s = protected_log(choice_probs_s[np.arange(session.n_trials), choices_s])
        session_log_likelihood = np.sum(trial_log_likelihood_f) + np.sum(trial_log_likelihood_s)  
        return session_log_likelihood

# -------------------------------------------------------------------------------------
# Q0 agent - original task version.
# -------------------------------------------------------------------------------------

class Random_first_step_orig():
    ''' Agent which makes random choice at first step.'''

    def __init__(self):

        self.name = 'Rand FS, original task.'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_s = np.zeros([2,2]) # Second step action values, indicies: [state, action]

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c   = randint(0,1)                        # First step action.
            s   = task.first_step(c)                  # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp)) # Second_step action.
            o   = task.second_step(s, c_s)            # Trial outcome.   

            # update action values. 
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes


