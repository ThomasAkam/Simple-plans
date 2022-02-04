import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import sem, ttest_rel
from .logistic_regression import COTI_model, CCOTI_model, lagged_model
from . import model_fitting as mf

plt.ion()
plt.rcParams['pdf.fonttype'] = 42

# -------------------------------------------------------------------------------------
# Utility functions.
# -------------------------------------------------------------------------------------

def exp_mov_ave(data, tau = 8., initValue = 0.5):
    'Exponential Moving average for 1d data.'
    m = np.exp(-1./tau)
    i = 1 - m
    mov_ave = np.zeros(np.size(data)+1)
    mov_ave[0] = initValue
    for k, sample in enumerate(data):
        mov_ave[k+1] = mov_ave[k] * m + i * sample 
    return mov_ave[1::]

def setup_figure(fig_no = 1, clf = True):
    plt.figure(fig_no)
    if clf:plt.clf()

# -------------------------------------------------------------------------------------
# Analysis functions.
# -------------------------------------------------------------------------------------

def session_stay_probs(session):
    'Evaluate stay probabilities for a single session'
    choices, transitions, second_steps, outcomes = session.CTSO_unpack(bool)
    stay = choices[1:] == choices[:-1]
    stay_probs = np.zeros(4)
    stay_probs[0] = np.mean(stay[ transitions[:-1] &  outcomes[:-1]]) # Rewarded, common transition.
    stay_probs[1] = np.mean(stay[~transitions[:-1] &  outcomes[:-1]]) # Rewarded, rare transition.
    stay_probs[2] = np.mean(stay[ transitions[:-1] & ~outcomes[:-1]]) # Non-rewarded, common transition.
    stay_probs[3] = np.mean(stay[~transitions[:-1] & ~outcomes[:-1]]) # Non-rewarded, rare transition.
    return stay_probs

def analyse_sessions(sessions):
    print('Analysing data: ' + sessions[0].name)
    data = {'stay_probs': np.array([s.stay_probs for s in sessions]),
            'name': sessions[0].name} 
    # Fitting.
    data['COTI_fits']   = mf.fit_sessions(sessions, COTI_model)
    data['CCOTI_fits']  = mf.fit_sessions(sessions, CCOTI_model)
    data['lagged_fits'] = mf.fit_sessions(sessions, lagged_model)
    return data

def likelihood_comparison(sessions, agents, fig_no = 1, 
                          sessions_to_compare = ['Q(1)', 'Q(0)', 'MB', 'RC', 'LS'],
                          agents_to_compare   = ['Q(1)', 'Q(0)', 'MB', 'RC', 'LS']):
    '''Evaluate likelihood for ML fits of agents specified in agents_to_compare to sessions
    simulated by agents specified in sessions_to_compare.'''
    session_likelihoods, session_BICs, p_values, p_values_BIC = ({},{},{},{})
    for sim_agent in sessions_to_compare:
        print('Evaluating likelihoods for ' + sim_agent + ' sessions.')
        session_likelihoods[sim_agent] = np.zeros([len(sessions[sim_agent]), len(agents_to_compare)])
        session_BICs[sim_agent]        = np.zeros([len(sessions[sim_agent]), len(agents_to_compare)])
        for i, fit_agent in enumerate(agents_to_compare):
            session_likelihoods[sim_agent][:,i] = [mf.fit_session(session, agents[fit_agent])['likelihood']
                                                   for session in sessions[sim_agent]]
            session_BICs[sim_agent][:,i] = -2 * session_likelihoods[sim_agent][:,i] + \
                            agents[fit_agent].n_params * np.log(sessions[sim_agent][0].n_trials)
        # Evaluate p values for differences between sim_agent and other agent likelihoods.
        sim_agent_likelihoods = session_likelihoods[sim_agent][:,agents_to_compare.index(sim_agent)]
        sim_agent_BICs = session_BICs[sim_agent][:,agents_to_compare.index(sim_agent)]
        p_values[sim_agent] = np.ones(len(agents_to_compare))
        p_values_BIC[sim_agent] = np.ones(len(agents_to_compare))
        for i, fit_agent in enumerate(agents_to_compare):
            if not fit_agent == sim_agent:
                p_values[sim_agent][i] = ttest_rel(session_likelihoods[sim_agent][:,i],
                                                   sim_agent_likelihoods)[1]
                p_values_BIC[sim_agent][i] = ttest_rel(session_BICs[sim_agent][:,i],
                                                   sim_agent_BICs)[1]
        likelihood_data = {'sessions_to_compare' : sessions_to_compare,
                           'agents_to_compare'   : agents_to_compare,
                           'session_likelihoods' : session_likelihoods,
                           'session_BICs'        : session_BICs,
                           'p_values'            : p_values,
                           'p_values_BIC'        : p_values_BIC}
    return likelihood_data

# -------------------------------------------------------------------------------------
# Plotting function.
# -------------------------------------------------------------------------------------

def session_plot(session, fig_no = 1):
    'Plot choices and reward probabilities for a single session.'
    choices, transitions, second_steps, outcomes = session.CTSO_unpack()
    choice_mov_ave = exp_mov_ave(choices)
    plt.figure(fig_no, figsize = [3.5,2])
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(choice_mov_ave, 'k.-', markersize = 3)
    plt.ylim(0,1)
    plt.ylabel('Choice moving average')
    plt.subplot(2,1,2)
    if len(session.reward_probs.shape) == 3:
        reward_probs = session.reward_probs.reshape([session.n_trials,4])
    else:
        reward_probs = session.reward_probs
    plt.plot(np.arange(session.n_trials), reward_probs)
    plt.ylim(0,1)
    plt.ylabel('Reward probs.')
    plt.xlabel('Trials')
    plt.yticks([0, 0.25, 0.5, 0.75, 1]) 
    print('Fraction of trials rewarded: {}'
           .format(sum(outcomes)/session.n_trials))


def stay_prob_plot(stay_probs, fig_no = 1, clf = True):
    'Plot stay probabilities with SEM errorbars for a list of sessions.'
    setup_figure(fig_no, clf)
    plt.bar(np.arange(1,5), np.mean(stay_probs,0), yerr = sem(stay_probs,0),
            error_kw = {'ecolor': 'r', 'capsize': 5, 'elinewidth': 5})
    plt.ylim(0,1)
    plt.xlim(0.75,5)
    plt.xticks([1.5,2.5,3.5,4.5],['1/Nr', '1/Ra', '0/Nr', '0/Ra'])
    plt.ylabel('Stay Probability')


def fit_plot(fits, fig_no = 1, clf = True, col = 'b'):
    'Plot model fit with sem errorbars.'
    setup_figure(fig_no, clf)
    x = np.arange(1, fits['n_params'] + 1)
    plt.errorbar(x, np.mean(fits['params'],0), sem(fits['params'],0),
                    linestyle = '', capsize = 5,  elinewidth = 3, color = col)
    plt.plot([0.5, fits['n_params'] + 0.5], [0, 0], 'k')
    plt.xticks(x, fits['param_names'])
    plt.xlim([0.5, fits['n_params'] + 0.5])


def lagged_fit_plot(fits, fig_no = 1, clf = True):
    'Plot model fit for logistic regression model with lagged predictors, with sem errorbars.'
    setup_figure(fig_no, clf)
    mean_params = np.mean(fits['params'],0)
    sem_params  = sem(fits['params'],0)
    param_lags = np.array([int(pn.split('-')[1]) for pn in fits['param_names']])
    param_base = [pn.split('-')[0] for pn in fits['param_names']]
    base_params = list(set(param_base))
    color_idx = np.linspace(0, 1, len(base_params))
    for i, base_param in zip(color_idx, base_params):
        p_mask = np.array([pb == base_param for pb in param_base])
        plt.errorbar(-param_lags[p_mask], mean_params[p_mask], sem_params[p_mask],
                     label = base_param, color = plt.cm.coolwarm(i))
    plt.plot([-max(param_lags) - 0.5, -0.5], [0,0], 'k')
    plt.xlim([-max(param_lags) - 0.5, -0.5])
    plt.xlabel('Lag (trials)')
    plt.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0., fontsize = 'small')


def sessions_data_plot(data, fig_no = 1):
    'Plot data from analysis of a set of sessions.'
    plt.figure(fig_no, figsize = [3.8,3.4])
    plt.clf()
    plt.subplot(2,2,1)
    stay_prob_plot(data['stay_probs'], fig_no, False)
    plt.subplot(2,2,2)
    fit_plot(data['COTI_fits'], fig_no, False)
    plt.ylim([-0.4, max([2.2, plt.ylim()[1]])])
    plt.title(data['name'])
    plt.subplot(2,2,3)
    fit_plot(data['CCOTI_fits'],fig_no, False)
    plt.ylim([-0.4, max([2.2, plt.ylim()[1]])])
    plt.subplot(2,2,4)
    lagged_fit_plot(data['lagged_fits'], fig_no, False)
    plt.ylim(min([-1, plt.ylim()[0]]), max([0.8, plt.ylim()[1]]))

def predictor_correlations(sessions,agent, fig_no = 1, cmap = 'BuPu'):
    ''' Evaluate and plot correlation matrix between predictors in 
    logistic regression models.
    '''
    predictors = []
    for session in sessions:
        predictors.append(agent._get_session_predictors(session))
    predictors = np.vstack(predictors)
    R = np.corrcoef(predictors.T)[:,::-1]
    n_params = agent.n_params
    plt.figure(fig_no)
    plt.clf()
    plt.pcolor(R, cmap = cmap, vmin = 0, vmax = 1)
    plt.colorbar()
    plt.xticks(np.arange(n_params)+0.5, agent.param_names[::-1])
    plt.yticks(np.arange(n_params)+0.5, agent.param_names)

def surface_plot(opt, fig_no = 1):
    'Plot fraction correct as function of parameter values'
    fig = plt.figure(fig_no)
    fig.clf()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(opt['X_grid_U'], opt['Y_grid_U'], opt['fun'], rstride = 1, cstride = 1,  
                           linewidth = 0, antialiased = False, cmap = plt.cm.coolwarm)
    plt.xlim(opt['X_grid_U'][0,0], opt['X_grid_U'][-1, 0])
    plt.ylim(opt['Y_grid_U'][0,0], opt['Y_grid_U'][ 0,-1])
    plt.xlabel(opt['param_names'][0])
    plt.ylabel(opt['param_names'][1])

def likelihood_comparison_plot(likelihood_data, fig_no = 1, use_BIC = False):
    plt.figure(fig_no, figsize = [7,1.2])
    plt.clf()
    nc = len(likelihood_data['agents_to_compare'])
    ns = len(likelihood_data['sessions_to_compare'])
    for i, sim_agent in enumerate(likelihood_data['sessions_to_compare']):
        if use_BIC:
            session_likelihoods = likelihood_data['session_BICs'][sim_agent]
            ylabel = 'BIC score'
        else:
            session_likelihoods = likelihood_data['session_likelihoods'][sim_agent]
            ylabel = 'Log. likelihood'
        plt.subplot(1, ns, i + 1)
        plt.errorbar(np.arange(1, nc + 1), np.mean(session_likelihoods,0), yerr = sem(session_likelihoods,0),
                    linestyle = '', color = 'b', capsize = 5,  elinewidth = 3)
        plt.xlim(0.5, nc + 0.5)
        plt.xticks(np.arange(1, nc + 1), likelihood_data['agents_to_compare'])

        plt.ylabel(ylabel)
        plt.title(sim_agent)

def performance_plot(performance, fig_no = 1, agent_order = None):
    if agent_order is None:
        agent_order = performance['agents_sorted']
    means_sorted = [performance[a_type]['mean'] for a_type in agent_order]
    SEMs_sorted  = [performance[a_type]['SEM']  for a_type in agent_order]
    plt.figure(fig_no, figsize = [2.5,2])
    plt.clf() 
    nc = len(agent_order)   
    plt.bar(np.arange(0.5, nc + 0.5), means_sorted, yerr = SEMs_sorted, color = 'k', 
            error_kw = {'ecolor': 'r', 'capsize': 5, 'elinewidth': 3})
    plt.xticks(np.arange(1, nc + 1), agent_order)
    plt.ylim(0.5,0.7)
    plt.xlim(0.25, nc + 0.5)
    plt.ylabel('Fraction of trials rewarded')

def action_values_plot(sessions, agent, fig_no = 1):
    'Plot values of chosen and non chosen actions as a function of trial type.'
    mean_Q_chosen             = np.zeros([len(sessions), 4])
    mean_Q_not_chosen         = np.zeros([len(sessions), 4])
    mean_Q_chosen_exploit     = np.zeros([len(sessions), 4])
    mean_Q_not_chosen_exploit = np.zeros([len(sessions), 4])
    mean_Q_chosen_explore     = np.zeros([len(sessions), 4])
    mean_Q_not_chosen_explore = np.zeros([len(sessions), 4])
    for i, session in enumerate(sessions):
        choices, transitions, second_steps, outcomes = session.CTSO_unpack(bool)
        Qs = agent.session_likelihood(session, agent.params, return_Qs = True)
        Q_chosen     = Qs[np.arange(len(choices)),    choices.astype(int)]
        Q_not_chosen = Qs[np.arange(len(choices)), 1- choices.astype(int)]
        rew_com  =  transitions &  outcomes
        rew_rare = ~transitions &  outcomes
        non_com  =  transitions & ~outcomes
        non_rare = ~transitions & ~outcomes
        exploit  = Q_chosen >= Q_not_chosen
        mean_Q_chosen[i,:] = [np.mean(Q_chosen[rew_com]), np.mean(Q_chosen[rew_rare]),
                              np.mean(Q_chosen[non_com]), np.mean(Q_chosen[non_rare])]
        mean_Q_not_chosen[i,:] = [np.mean(Q_not_chosen[rew_com]), np.mean(Q_not_chosen[rew_rare]),
                                  np.mean(Q_not_chosen[non_com]), np.mean(Q_not_chosen[non_rare])]
        mean_Q_chosen_exploit[i,:] = [np.mean(Q_chosen[rew_com & exploit]), np.mean(Q_chosen[rew_rare & exploit]),
                                      np.mean(Q_chosen[non_com & exploit]), np.mean(Q_chosen[non_rare & exploit])]
        mean_Q_not_chosen_exploit[i,:] = [np.mean(Q_not_chosen[rew_com & exploit]), np.mean(Q_not_chosen[rew_rare & exploit]),
                                          np.mean(Q_not_chosen[non_com & exploit]), np.mean(Q_not_chosen[non_rare & exploit])]
        mean_Q_chosen_explore[i,:] = [np.mean(Q_chosen[rew_com & ~exploit]), np.mean(Q_chosen[rew_rare & ~exploit]),
                                      np.mean(Q_chosen[non_com & ~exploit]), np.mean(Q_chosen[non_rare & ~exploit])]
        mean_Q_not_chosen_explore[i,:] = [np.mean(Q_not_chosen[rew_com & ~exploit]), np.mean(Q_not_chosen[rew_rare & ~exploit]),
                                          np.mean(Q_not_chosen[non_com & ~exploit]), np.mean(Q_not_chosen[non_rare & ~exploit])]
   
    def av_plot(mean_Q_chosen, mean_Q_not_chosen):
        ekw = {'ecolor': 'y', 'capsize': 3, 'elinewidth': 3} 
        b1 = plt.bar(np.arange(4), np.mean(mean_Q_chosen,0), 0.35, color = 'b',
                     yerr = sem(mean_Q_chosen,0), error_kw = ekw)
        b2 = plt.bar(np.arange(4) + 0.35, np.mean(mean_Q_not_chosen,0), 0.35, color = 'r',
                     yerr = sem(mean_Q_not_chosen,0), error_kw = ekw)
        plt.xlim(-0.35,4)
        plt.xticks(np.arange(4) + 0.35,['Rew com.', 'Rew rare', 'Non com.', 'Non rare'])
        return b1, b2
                                         
    plt.figure(fig_no)
    plt.clf()
    plt.subplot(1,3,1)
    av_plot(mean_Q_chosen, mean_Q_not_chosen)
    plt.ylabel('Action value')
    plt.title('All trials')
    plt.subplot(1,3,2)
    av_plot(mean_Q_chosen_exploit, mean_Q_not_chosen_exploit)    
    plt.title('Exploit trials')
    plt.subplot(1,3,3)
    b1, b2 = av_plot(mean_Q_chosen_explore, mean_Q_not_chosen_explore) 
    plt.title('Explore trials')
    plt.legend( (b1[0], b2[0]), ('Chosen', 'Not chosen'), bbox_to_anchor=(1, 2))

def multi_agent_log_reg_surface_plot(grid_sims, fig_no = 1, vmax = None, cmap = 'BuPu', predictor = 'trans_x_out'):
    '''Plot logistic regression predictor loadings as a function of agent parameter values
    for a set of agents using the same z axis scaleing.'''
    min_loadings, max_loadings = ([],[])
    for a_type in grid_sims.keys():
        mean_loadings = _mean_predictor_loadings(grid_sims[a_type], predictor)
        min_loadings.append(np.min(mean_loadings))
        max_loadings.append(np.max(mean_loadings))
    if vmax:
        vrange = (0, vmax)
    else:    
        vrange = (min(min_loadings), max(max_loadings))
    f = plt.figure(fig_no, figsize = [16,2.4])
    plt.clf()
    n_agents = len(grid_sims.keys())
    for i, a_type in enumerate(grid_sims.keys()):
        plt.subplot(1, n_agents, i + 1)
        _log_reg_surface_plot(grid_sims[a_type], predictor, vrange, cmap)
        plt.title(a_type)
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.1, 0.01, 0.8])
    plt.colorbar(cax=cbar_ax)

def _log_reg_surface_plot(grid_sim, predictor = 'trans_x_out', vrange = None, cmap = 'BuPu'):
    'Plot logistic regression predictor loading as function of agent parameter values.'
    mean_loadings = _mean_predictor_loadings(grid_sim, predictor)
    grid_N = grid_sim.shape[0]
    if vrange:
        plt.pcolor(mean_loadings, vmin = vrange[0], vmax = vrange[1], cmap = cmap)
    else:
        plt.pcolor(mean_loadings, cmap = cmap)
    plt.xticks(np.arange(grid_N) + 0.5, np.round(grid_sim[0,0]['param_grid'][1],2))
    plt.yticks(np.arange(grid_N) + 0.5, np.round(grid_sim[0,0]['param_grid'][0],2))
    plt.xlabel(grid_sim[0,0]['param_names'][1])
    plt.ylabel(grid_sim[0,0]['param_names'][0])


def _mean_predictor_loadings(grid_sim, predictor):
    n_sessions = len(grid_sim[0,0]['sessions'])
    grid_N = grid_sim.shape[0]
    pred_ind = grid_sim[0,0]['COTI_fit']['param_names'].index(predictor)
    session_predictor_loadings = np.zeros([grid_N, grid_N, n_sessions])
    for i in range(grid_N):
        for j in range(grid_N):
            session_predictor_loadings[i,j,:] = grid_sim[i,j]['COTI_fit']['params'][:,pred_ind]
    return np.mean(session_predictor_loadings,2)





