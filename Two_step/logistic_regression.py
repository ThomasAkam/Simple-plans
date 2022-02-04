import numpy as np
from .RL_agents import protected_log
import sys

log_max_float = np.log(sys.float_info.max) # Log of largest possible floating point number.

# -------------------------------------------------------------------------------------
# Utility functions.
# -------------------------------------------------------------------------------------

def _protected_products(x,a):
    ax  = a*x
    ax[ax > log_max_float] = log_max_float
    _ax = -a*x
    _ax[_ax > log_max_float] = log_max_float
    return (ax, _ax)

def smooth_l1(x, a):
    'smooth aproximation to the L1 norm of x.'
    ax, _ax = _protected_products(x, a)
    smooth_abs = (1./a)*(np.log(1 + np.exp(_ax)) + np.log(1 + np.exp(ax)))
    return np.sum(smooth_abs)

def smooth_l1_grad(x,a):
    'Grad of smooth aproximation to the L1 norm.'
    ax, _ax = _protected_products(x, a)
    return (1./(1. + np.exp(_ax)) - 1./(1. + np.exp(ax)))


# -------------------------------------------------------------------------------------
# Configurable logistic regression model.
# -------------------------------------------------------------------------------------

class config_log_reg():

    '''
    Configurable logistic regression agent. Arguments:

    predictors - The basic set of predictors used is specified with predictors argument.  

    lags         - By default each predictor is only used at a lag of -1 (i.e. one trial predicting the next).
                 The lags argument is used to specify the use of additional lags for specific predictors:
                 e.g. lags = {'outcome': 3, 'choice':2} specifies that the outcomes on the previous 3 trials
                 should be used as predictors, while the choices on the previous 2 trials should be used.
                 lags can also be an int, in which case all predictors are given the specified number of lags.

    '''


    def __init__(self, base_predictors = ['correct', 'choice','outcome','transition', 'trans_x_out'],
                lags = {}, reg_strength = 0):

        self.name = 'config_lr'
        self.base_predictors = base_predictors # Predictor names ignoring lags.
        self.reg_strength = reg_strength # Regularisation strength.

        self.predictors = [] # Predictor names including lags.
        if type(lags) == int:
            lags = dict(zip(base_predictors,np.ones(len(base_predictors), int) * lags))
        for predictor in self.base_predictors:
            if predictor in lags.keys():
                for i in range(lags[predictor]):
                    self.predictors.append(predictor + '-' + str(i + 1)) # Lag is indicated by value after '-' in name.
            else:
                self.predictors.append(predictor) # If no lag specified, defaults to 1.

        self.n_params = len(self.predictors)
        self.param_names = self.predictors

    def _get_session_predictors(self, session):
        
        '''Calculate and return values of predictor variables for all trials in session.
        '''
        # Evaluate base (non-lagged) predictors from session events.

        choices, transitions, second_steps, outcomes  = session.CTSO_unpack(bool)

        bp_values = {} 

        for p in self.base_predictors:

            if p in ('correct', 'correct_cont'):  
                if len(session.reward_probs.shape) == 3: # Version of task with choice at second step.
                    rp = np.max(session.reward_probs, 2) # Take max over actions in each state.
                else:
                    rp = session.reward_probs
                if p == 'correct':  # Binary valued correct predictor.
                    correct = np.zeros(session.n_trials)  
                    correct[rp[:,1] > rp[:,0]] =  0.5
                    correct[rp[:,0] > rp[:,1]] = -0.5 
                elif p == 'correct_cont': # Continous valued correct predictor.
                    correct = rp[:,1] - rp[:,0] 
                bp_values[p] = correct  # 0.5, 0, -0.5 for action 1 being correct, neutral, incorrect option.

            elif p ==  'choice': # Previous choice predicts current choice (0.5, -0.5)
                bp_values[p] = choices - 0.5

            elif p ==  'outcome': # Reward predicts repeating choice (0.5, - 0.5)
                bp_values[p] = (outcomes == choices) - 0.5

            elif p ==  'transition': # Common transition predicts repeating choice (0.5, -0.5)    
                    bp_values[p] = (transitions == choices)  - 0.5

            elif p == 'trans_x_out': # Transition-outcome interaction predicts repeating choice (0.5, -0.5)
                    bp_values[p] = ((transitions == outcomes)  == choices) - 0.5

            elif p == 'rew_com':  # Rewarded common transition predicts repeating choice.
                bp_values[p] = ( outcomes &  transitions) * (choices - 0.5)

            elif p == 'rew_rare':  # Rewarded rare transition predicts repeating choice.
                bp_values[p] = ( outcomes & ~transitions) * (choices - 0.5)   

            elif p == 'non_com':  # Non-rewarded common transition predicts repeating choice.
                bp_values[p] = (~outcomes &  transitions) * (choices - 0.5)

            elif p == 'non_rare':  # Non-Rewarded rare transition predicts repeating choice.
                bp_values[p] = (~outcomes & ~transitions) * (choices - 0.5)

            elif p == 'bias':  # Constant bias.
                bp_values[p] = np.ones(len(choices))             

        # Generate lagged predictors from base predictors.

        session_predictors = np.zeros([session.n_trials, self.n_params])

        for i,p in enumerate(self.predictors):  
            if '-' in p: # Get lag from predictor name.
                lag = int(p.split('-')[1]) 
                bp_name = p.split('-')[0]
            else:        # Use default lag.
                lag = 1
                bp_name = p
            session_predictors[lag:, i] = bp_values[bp_name][:-lag]

        return session_predictors


    def session_likelihood(self, session, weights, eval_grad = True, sign = 1.):

        choices = session.CTSO['choices']

        session_predictors = self._get_session_predictors(session) # Get array of predictors

        # Evaluate session log likelihood.

        Q  = np.dot(session_predictors,weights)
        Q[Q < -log_max_float] = -log_max_float # Protect against overflow in exponential. 
        P  = 1./(1. + np.exp(-Q))  # Probability of making choice 1
        Pc = 1 - P - choices + 2. * choices * P  

        session_log_likelihood = sum(protected_log(Pc)) 

        if self.reg_strength:
            session_log_likelihood -= self.reg_strength * smooth_l1(weights, 100) 

        # Evaluate session log likelihood gradient.

        if eval_grad:
            dLdQ  = - 1 + 2 * choices + Pc - 2 * choices * Pc
            dLdW = sum(np.tile(dLdQ,(len(weights),1)).T * session_predictors, 0) # Likelihood gradient w.r.t weights.
            session_log_likelihood_gradient = dLdW
            if self.reg_strength:
                session_log_likelihood_gradient -= self.reg_strength * smooth_l1_grad(weights, 100)
            return (sign * session_log_likelihood, sign * session_log_likelihood_gradient)
        else:
            return sign * session_log_likelihood


# -------------------------------------------------------------------------------------
# Model instances.
# -------------------------------------------------------------------------------------


COTI_model = config_log_reg(['choice','outcome','transition', 'trans_x_out'])

CCOTI_model = config_log_reg(['correct', 'choice','outcome','transition', 'trans_x_out'])

lagged_model = config_log_reg(['rew_com', 'rew_rare', 'non_com', 'non_rare'], lags = 10)

CcCOTI_model = config_log_reg(['correct_cont', 'choice','outcome','transition', 'trans_x_out'])


