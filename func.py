# from HCRP_LM.ddHCRP_LM import *
from ddHCRP_LM import *
import _pickle as cPickle
import os, sys, inspect, platform
import psutil
import seaborn as sns
# from plot_mm import plot_mm
import copy
import collections
import glob
from collections import defaultdict

if 'Windows' in platform.platform():
    slash = '\\'
else:
    slash = '/'

cwd = os.getcwd()

def generate_bouts_openloop(m, n_bouts=100):

    bouts       = []
    pred_probs  = []
    occurrences = []
    t_bouts     = []
    bout_indeces= []
    for n in range(n_bouts):

        print(n)

        next_element = 'START'
        bout        = [next_element]
        pred_prob   = [np.nan]
        occurrence  = [np.nan]
        syl_occurrences = defaultdict(lambda: 1)
        while next_element!='STOP':

            u = bout[min(0, -m.n + 1):]
            next_element = m.predict_next_word(t=0, u=u)
            if next_element=='START':
                continue
            p_next_element = m.word_probability_all_samples(t=0, u=u, w=next_element)
            o_next_element = syl_occurrences[next_element]

            syl_occurrences[next_element] += 1

            bout.append(next_element)
            pred_prob.append(p_next_element)
            occurrence.append(o_next_element)

        t_bout = [np.nan] + list(range(1, len(bout)-1)) + [np.nan]
        bout_index = [len(bouts)]*len(bout)

        bouts.append(bout)
        pred_probs.append(pred_prob)
        occurrences.append(occurrence)
        t_bouts.append(t_bout)
        bout_indeces.append(bout_index)

    df = pd.DataFrame({'bout_index'  : flatten(bout_indeces),
                        't_bout'    : flatten(t_bouts),
                        'occurrence': flatten(occurrences),
                        'syllable'  : flatten(bouts),
                        'pred_prob' : flatten(pred_probs)}
                        )

    return df


def get_data(experimenter='Lena', phase='baseline'):

    data = {}
    if experimenter=='Lena':
        subjects = ['bu86bu48','gr54bu78', 'gr57bu40', 'gr58bu60', 'rd82wh13', 'rd49rd79', 'wh08pk40', 'wh09pk88']
    if experimenter=='Simon':
        subjects = ['rd6030', 'rd8031', 'rd6bu6', 'ye0wh0', 'rd5374']

    if phase=='baseline':

        for subject in subjects:
            data_label = subject
            data[data_label] = []
            with open(f'{cwd}{slash}data{slash}baseline_data{slash}{subject}.txt') as f:
                contents = f.read()
            # we skip first and last separator because they produce empty strings

            if experimenter=='Lena':
                for bout in contents.split('Y')[1:-1]:
                    data[data_label].append(['START'] + list(bout) + ['STOP'])
            if experimenter=='Simon':
                for bout in contents.split(','):
                    data[data_label].append(['START'] + list(bout) + ['STOP'])

    elif phase=='post-training':

        cwd = os.getcwd()

        for subject in subjects:
            for target in ['T1','T2']:
                data_label = f'{subject}_{target}'
                data[data_label] = []
                with open(f'{cwd}{slash}data{slash}raw_posttraining_data{slash}{subject}_{target}.txt') as f:
                    contents = f.read()
                # we skip first and last separator because they produce empty strings

                if experimenter=='Lena':
                    for bout in contents.split('Y')[1:-1]:
                        data[data_label].append(['START'] + list(bout) + ['STOP'])
                if experimenter=='Simon':
                    # To be implemented if we get the post-training raw data from him too
                    return

    elif phase=='synthetic':

        cwd = os.getcwd()
        df = pd.read_csv(f'{cwd}{slash}data{slash}synthetic_bouts.csv')

        for subject in subjects:
            data_label = subject
            data[data_label] = []
            subdf = df[df['subject']==subject]

            for bout_index in subdf.bout_index.unique():
                bout = subdf[subdf['bout_index']==bout_index].syllable.values
                data[data_label].append(['START'] + list(bout) + ['STOP'])

    return data

def get_timestamped_data(phase='baseline'):

    data = {}
    onsets = {}
    durations = {}
    latencies = {}
    subjects = ['bu86bu48','gr54bu78', 'gr57bu40', 'gr58bu60', 'rd82wh13', 'rd49rd79', 'wh08pk40', 'wh09pk88']

    if phase=='baseline':

        for subject in subjects:
            data_label = subject
            data[data_label] = []
            onsets[data_label] = []
            durations[data_label] = []
            latencies[data_label] = []

            directory = f'{cwd}{slash}data{slash}timestamped_data{slash}{subject}{slash}baseline'

            csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]  # get list of CSV files
            for f in csv_files:
                df = pd.read_csv(os.path.join(directory, f))
                bout = [x.split('.')[0] for x in df.columns]

                data[data_label].append(['START'] + bout + ['STOP'])
                start = df.iloc[0].values
                end = df.iloc[1].values

                # set everything relative to the first onset (because sometimes there is a long quiet latency before the first syllable in the bout)
                end   = end - start[0]
                start = start - start[0]

                onsets[data_label].append([np.nan]            + list(start)             + [np.nan])
                durations[data_label].append([np.nan]         + list(end-start)             + [np.nan])
                latencies[data_label].append([np.nan, np.nan] + list(start[1:] - end[:-1])  + [np.nan])

    elif phase=='post-training':
        # TODO implement
        return True

    return data, onsets, durations, latencies


def get_joint_timestamped_data(include_durations=True, include_latencies=True):

    # this will discretize the durations and latencies into 10ms bins from 0 to 200ms;
    # TODO resolution could be a free parameter
    bins = np.linspace(0, 200, 41)
    len_bins = len(bins)

    all_data, all_onsets, all_durations, all_latencies = get_timestamped_data(phase='baseline')

    joint_timestamped_data = {}
    for subject in all_data.keys():

        joint_timestamped_data[subject] = []

        for i_bout in range(len(all_data[subject])):

            syllables = all_data[subject][i_bout]
            durations = all_durations[subject][i_bout]
            latencies = all_latencies[subject][i_bout]

            durations_bins = np.digitize(durations, bins)
            durations_digitized = [int(bins[i-1]) if i<len(bins) else np.nan for i in durations_bins]

            latencies_bins = np.digitize(latencies, bins)
            latencies_digitized = [int(bins[i-1]) if i<len(bins) else np.nan for i in latencies_bins]

            if not include_durations and not include_latencies:
                x = [str((syllables[i_syllable])) for i_syllable in range(len(syllables))]

            elif include_durations and not include_latencies:
                x = [str((syllables[i_syllable], durations_digitized[i_syllable])) for i_syllable in range(len(syllables))]

            elif include_latencies and not include_durations:
                x = [str((syllables[i_syllable], latencies_digitized[i_syllable])) for i_syllable in range(len(syllables))]

            else:
                x = [str((syllables[i_syllable], durations_digitized[i_syllable], latencies_digitized[i_syllable])) for i_syllable in range(len(syllables))]

            joint_timestamped_data[subject].append(x)

    return joint_timestamped_data


def create_timestamped_dataframe():
    all_data, all_onsets, all_durations, all_latencies = get_timestamped_data(phase='baseline')
    n_bouts = 10000  # max number of bouts per bird to include in analyses; set to 10000 if you want all
    timestamp_df = pd.DataFrame()

    for subject in all_data.keys():

        print(subject)

        with open(f"{subject}_deepmodel.pickle", "rb") as handle:
            m = cPickle.load(handle)

        m.predict(corpus_segments=all_data[subject][:n_bouts])

        m_firstorder = HCRP_LM([1,1])
        m_firstorder.fit(corpus_segments=all_data[subject][:n_bouts], frozen=True)

        durations = flatten(all_durations[subject][:n_bouts])
        latencies = flatten(all_latencies[subject][:n_bouts])
        bouts     = flatten(all_data[subject][:n_bouts])

        df = pd.DataFrame({'subject'                : [subject]*len(bouts),
                            'syllable'              : bouts,
                            'duration'         : durations,
                            'latency'          : latencies,
                            'contextual_likelihood' : m.choice_probs,
                            'firstorder_contextual_likelihood' : m_firstorder.choice_probs})

        timestamp_df = timestamp_df.append(df)

    return timestamp_df


def plot_prediction(corpus, dishes, predictive_distr, context_importance, essential_context_depths=None, start=None, end=None):

    f, ax = plt.subplots(2, 1, figsize=((end-start)*.20, 10))

    sns.heatmap(predictive_distr[start:end].T, vmin=0, vmax=1, cmap='Greys', ax=ax[0], cbar_kws={'label': r'$p$(syllable)', 'ticks':[0,1]})
    for syllable_i, syllable in enumerate(corpus[start:end]):
        ax[0].scatter(x=syllable_i+0.5, y=dishes.index(syllable)+0.5, marker='X', c='red')

    xticklabels = [x if x not in ['START','STOP'] else '!' for x in corpus[start:end]]
    ax[0].set_yticks(np.array(range(len(dishes)))+0.5)
    ax[0].set_yticklabels(dishes, rotation=0)
    ax[0].set_ylabel('vocabulary')
    ax[0].set_title('Syllable prediction')
    ax[0].set_xticks(np.array(range(len(corpus[start:end])))+0.5)
    ax[0].set_xticklabels(xticklabels, rotation=0)

    sns.heatmap(context_importance[start:end].T, cmap='Reds', ax=ax[1], cbar_kws={'label': 'KL divergence\nfrom full prediction'}, vmin=0)  # r'$\Delta$KL'
    if essential_context_depths is not None:
        for syllable_i, syllable in enumerate(corpus[start:end]):
            ax[1].scatter(x=syllable_i+0.5, y=essential_context_depths[start:end][syllable_i]+1, marker='_', c='k', s=100)
    ax[1].set_title('Essential context depth')
    ax[1].set_ylabel('context depth\n('+r'$n$ previous syllables)')
    ax[1].set_yticks(np.array([0,2,4,8,16,24])+0.5)
    ax[1].set_yticklabels([0,2,4,8,16,24])
    ax[1].set_xticks(np.array(range(len(corpus[start:end])))+0.5)
    ax[1].set_xticklabels(xticklabels, rotation=0)
    ax[1].set_xlabel('song bout (syllables)')

    return f

def get_bout_borders(bouts):
    bout_borders = []
    start = 0
    for bout in bouts:
        bout_borders.append((start, start+len(bout)))
        start+=len(bout)
    return bout_borders

def get_trialpredictions(data, subject, target='', fitted_model=None, model_order=None, alpha=1, essential_context_depth_threshold=5, save_trialpredictions=False, plot_example_bouts=False, test=False, output_label='test', print_progress=False):

    if target=='':
        bouts = data[f'{subject}']
    else:
        bouts = data[f'{subject}_{target}']
    all_flattened_bouts = [syl for bout in bouts for syl in bout]
    dishes = sorted(list(set(all_flattened_bouts)))

    bout_borders = get_bout_borders(bouts)

    if model_order is not None:
        m = HCRP_LM(strength=[alpha]*model_order, dishes=dishes)
        m.fit(corpus_segments=bouts, frozen=True, online_predict=True, compute_context_importance=True, compute_seat_odds=True)

    else:
        with open(f"{subject}_deepmodel.pickle", "rb") as handle:
           m = cPickle.load(handle)

    if test:
        m.predict(corpus_segments=bouts, compute_context_importance=True, compute_seat_odds=True)
        if print_progress:
            print('Test inference done.')
        with open(f"{subject}_{target}_deepmodel_{output_label}.pickle", "wb") as output_file:
           cPickle.dump(m, output_file)

    bout_indeces              = []
    t_bout                    = []
    occurrence                = []
    bouts_without_demarkation = []
    essential_context_depth   = []
    consistent_context_depth  = []
    pred_distr                = []
    event_prob                = []
    syllables, counts         = np.unique(all_flattened_bouts, return_counts=True)
    rare_syllables            = counts<len(all_flattened_bouts)/100
    syllables                 = syllables[~rare_syllables]
    syllables                 = [syl for syl in syllables if syl not in ['START', 'STOP']]
    syllable_pred_prob                  = dict([(syl,[]) for syl in syllables])
    syllable_essential_context_depth    = dict([(syl,[]) for syl in syllables])
    syllable_consistent_context_depth   = dict([(syl,[]) for syl in syllables])

    context_importance = m.context_importance
    # We could choose to use context gain, that is, the drop of KL due to adding more context, instead of the raw KL
    # KL_from_uniform = np.zeros(m.predictive_distr.shape[0])
    # for i in range(len(m.predictive_distr)):
    #     KL_from_uniform[i] = scipy.stats.entropy(m.predictive_distr[i], [0.25]*len(m.predictive_distr[i]))
    # context_gain = np.zeros(context_importance.shape)
    # context_gain[:, 0] = KL_from_uniform - context_importance[:,0]
    # for context_len in range(1,context_importance.shape[1]):
    #     context_gain[:, context_len] = context_importance[:, context_len-1] - context_importance[:, context_len]

    for bout_i, bout in enumerate(bouts):

        start=bout_borders[bout_i][0]
        end=bout_borders[bout_i][1]

        bout_indeces.append([bout_i]*len(bout[1:-1]))
        t_bout.append(list(range(1, len(bout[1:-1])+1)))

        bout_occurrence = []
        for syl_i, syl in enumerate(bout[1:-1]):
            bout_occurrence.append(len([s for s in bout[1:syl_i+1] if s==syl])+1)
        occurrence.append(bout_occurrence)
        bouts_without_demarkation.append(bout[1:-1])

        essential_context_depths = np.argmin(m.context_importance>essential_context_depth_threshold, axis=1)
        essential_context_depth.append(essential_context_depths[start+1:end-1])

        consistent_context_depths = np.argmax(m.seat_odds, axis=1)
        consistent_context_depth.append(consistent_context_depths[start+1:end-1])

        pred_distr.append(np.round(m.predictive_distr[start+1:end-1], 2))
        event_prob.append(np.round(m.choice_probs[start+1:end-1], 2))

        for syl_i, syl in enumerate(bout[1:-1]):
            # print(bout)
            if syl in syllables:
                syllable_pred_prob[syl].append(event_prob[-1][syl_i])
                syllable_essential_context_depth[syl].append(essential_context_depth[-1][syl_i])
                syllable_consistent_context_depth[syl].append(consistent_context_depth[-1][syl_i])

    d = pd.DataFrame([flatten(bout_indeces), flatten(t_bout), flatten(occurrence), flatten(bouts_without_demarkation), flatten(event_prob), flatten(essential_context_depth)]).T
    d.columns = ['bout_index', 't_bout', 'occurrence', 'syllable', 'pred_prob', 'essential_context_depth']
    if save_trialpredictions:
        if test:
            d.to_csv(f'{subject}_{target}_trialpredictions_deepmodel_{output_label}.csv', sep=';')
        else:
            if model_order is not None:
                d.to_csv(f'{subject}_trialpredictions_{str(model_order)}level.csv', sep=';')
            else:
                d.to_csv(f'{subject}_trialpredictions_deepmodel.csv', sep=';')

    # f,ax=plt.subplots(1, len(syllables), figsize=(len(syllables)*2, 2))
    # for syl_i, syl in enumerate(syllables):
    #     ax[syl_i].hist(syllable_essential_context_depth[syl])
    #     ax[syl_i].set_title(syl)
    # plt.suptitle(subject)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # path = cwd+'\\'+subject+"\\"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # plt.savefig(path+subject+'_syl_context_depth_deepmodel.png', transparent=False)  #, dpi=600
    # plt.close()

    median_syllable_pred_prob                = [np.median(syllable_pred_prob[syl]) for syl in syllables]
    median_syllable_essential_context_depth  = [np.median(syllable_essential_context_depth[syl]) for syl in syllables]
    median_syllable_consistent_context_depth = [np.median(syllable_consistent_context_depth[syl]) for syl in syllables]

    # syllable_stats = pd.DataFrame([syllables, median_syllable_pred_prob, median_syllable_essential_context_depth]).T
    # syllable_stats.columns = ['syllable', 'median_syllable_pred_prob', 'median_essential_context_depth']
    # syllable_stats.to_csv(subject + '_median_predictions_deepmodel.csv', sep=';')

    if plot_example_bouts:

        cwd = os.getcwd()

        for bout_i in list(range(0, len(bouts), 10)):
            print(bout_i)
            f = plot_prediction(corpus=all_flattened_bouts,
                             dishes=dishes,
                             predictive_distr=m.predictive_distr,
                             # context_importance=context_gain,
                             context_importance=m.context_importance,
                             essential_context_depths = essential_context_depths,
                             # context_importance=m.seat_odds,
                             # context_importance=m.seat_odds/m.seat_odds.sum(axis=1)[:,np.newaxis],  # normalized seat odds
                             start=bout_borders[bout_i][0],
                             end=bout_borders[bout_i][1]
                             )
            # f.suptitle('Bird ' + str(subject))
            f.tight_layout()
            path = cwd+'\\'+subject+"\\"
            if not os.path.exists(path):
                os.makedirs(path)
            f.savefig(f'{path}_{str(bout_i)}_{target}_KLdiv_deepmodel_{output_label}.png', transparent=False)  #, dpi=600
            plt.close()
    #
    #     f = plot_prediction(corpus=all_flattened_bouts,
    #                      dishes=dishes,
    #                      predictive_distr=m.predictive_distr,
    #                      # context_importance=m.context_importance,
    #                      # context_importance=m.seat_odds,
    #                      context_importance=m.seat_odds/m.seat_odds.sum(axis=1)[:,np.newaxis],  # normalized seat odds
    #                      start=bout_borders[bout_i][0]+2,
    #                      end=bout_borders[bout_i][1]
    #                      )
    #     # f.suptitle('Bird ' + str(subject))
    #     f.tight_layout()
    #     path = cwd+'\\'+subject+"\\"
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     f.savefig(path+ str(bout_i)+'_seatodds_deepmodel.png', transparent=False)  #, dpi=600
    #     plt.close()

    return d


def get_likelihoods_depths_of_targeted_syllables(data, model_order=None, alpha=1):

    # Filter for retrained transitions
    median_context_depths = {}
    mean_likelihoods = {}
    bigram_probabilities = {}
    for subject in data.keys():
        context_depth_target1 = []
        context_depth_target2 = []
        likelihood_target1 = []
        likelihood_target2 = []

        if model_order is not None:
            d = get_trialpredictions(data=data, subject=subject, model_order=model_order, alpha=alpha)
            print(f'Computed likelihoods of targeted syllables for subject {subject}; model order {model_order}', flush=True)
        else:
            d = pd.read_csv(subject + '_trialpredictions_deepmodel.csv', sep=';')

        for i,r in d.iloc[1:-1].iterrows():
            current     = r['syllable']
            previous    = d.iloc[i-1]['syllable']
            next        = d.iloc[i+1]['syllable']

            ############################ Lena sample ###########################

            if subject == 'bu86bu48':
                if (r['syllable']=='x' and previous == 'c'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='r' and previous == 'c'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            if subject == 'wh08pk40':
                if (r['syllable']=='c' and previous == 'b'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='c' and previous == 'e'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            if subject == 'wh09pk88':
                if (r['syllable']=='a' and previous =='f'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='a' and previous =='n'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            if subject == 'rd49rd79':
                if (r['syllable']=='b' and previous == 'a'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='x' and previous == 'd'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            if subject == 'gr57bu40':
                if (r['syllable']=='b' and previous == 'c'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='d' and previous == 'd'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            if subject == 'gr58bu60':
                if (r['syllable']=='b' and previous == 'a'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='d' and previous == 'a'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            if subject == 'gr54bu78':
                if (r['syllable']=='e' and previous == 'e'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='a' and previous == 'e'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            if subject == 'rd82wh13':
                if (r['syllable']=='d' and previous == 'l'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='e' and previous == 'l'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            ########################### Simon sample ###########################

            if subject == 'ye0wh0':
                if (r['syllable']=='a' and previous == 'b'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='e' and previous == 'b'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            if subject == 'rd6bu6':
                if (r['syllable']=='c' and previous == 'k'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='j' and previous == 'k'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            if subject == 'rd6030':
                if (r['syllable']=='b' and previous == 'y'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='h' and previous == 'y'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            if subject == 'rd8031':
                if (r['syllable']=='l' and previous == 'a'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='g' and previous == 'a'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

            if subject == 'rd5374':
                if (r['syllable']=='u' and previous == 'b'):
                    context_depth_target1.append(r['essential_context_depth'])
                    likelihood_target1.append(r['pred_prob'])
                elif (r['syllable']=='j' and previous == 'b'):
                    context_depth_target2.append(r['essential_context_depth'])
                    likelihood_target2.append(r['pred_prob'])

        ############################ Lena sample ###########################

        if subject == 'bu86bu48':
            frequency_target1 = ''.join(d['syllable']).count('cx')
            frequency_target2 = ''.join(d['syllable']).count('cr')

        if subject == 'wh08pk40':
            frequency_target1 = ''.join(d['syllable']).count('bc')
            frequency_target2 = ''.join(d['syllable']).count('ec')

        if subject == 'wh09pk88':
            frequency_target1 = ''.join(d['syllable']).count('fa')
            frequency_target2 = ''.join(d['syllable']).count('na')

        if subject == 'rd49rd79':
            frequency_target1 = ''.join(d['syllable']).count('ab')
            frequency_target2 = ''.join(d['syllable']).count('dx')

        if subject == 'gr57bu40':
            frequency_target1 = ''.join(d['syllable']).count('cb')
            frequency_target2 = ''.join(d['syllable']).count('dd')

        if subject == 'gr58bu60':
            frequency_target1 = ''.join(d['syllable']).count('ab')
            frequency_target2 = ''.join(d['syllable']).count('ad')

        if subject == 'gr54bu78':
            frequency_target1 = ''.join(d['syllable']).count('ee')
            frequency_target2 = ''.join(d['syllable']).count('ea')

        if subject == 'rd82wh13':
            frequency_target1 = ''.join(d['syllable']).count('ld')
            frequency_target2 = ''.join(d['syllable']).count('le')

        ########################### Simon sample ###########################

        if subject == 'ye0wh0':
            frequency_target1 = ''.join(d['syllable']).count('ba')
            frequency_target2 = ''.join(d['syllable']).count('be')

        if subject == 'rd6bu6':
            frequency_target1 = ''.join(d['syllable']).count('kc')
            frequency_target2 = ''.join(d['syllable']).count('kj')

        if subject == 'rd6030':
            frequency_target1 = ''.join(d['syllable']).count('yb')
            frequency_target2 = ''.join(d['syllable']).count('yh')

        if subject == 'rd8031':
            frequency_target1 = ''.join(d['syllable']).count('al')
            frequency_target2 = ''.join(d['syllable']).count('ag')

        if subject == 'rd5374':
            frequency_target1 = ''.join(d['syllable']).count('bu')
            frequency_target2 = ''.join(d['syllable']).count('bj')

        median_context_depths[subject]  = [np.median(context_depth_target1), np.median(context_depth_target2)]
        mean_likelihoods[subject]       = [np.mean(likelihood_target1), np.mean(likelihood_target2)]
        bigram_probabilities[subject]  = [frequency_target1/len(d), frequency_target2/len(d)]

    return mean_likelihoods, median_context_depths, bigram_probabilities

def get_training_score_table(mean_likelihoods, median_context_depths, bigram_probabilities, scores_file='Lena_learning_scores.csv'):

    d = pd.read_csv(scores_file, sep=';')

    d = d.T.iloc[2:]
    index = pd.MultiIndex.from_tuples([('baseline', 'C1'), ('baseline', 'C2'), ('training', 'C1'), ('training', 'C2')], names=["phase", "context"])
    d.columns = index

    d_target1 = pd.DataFrame(index=d.index)
    d_target1['context'] = ['C1']*len(d_target1)
    d_target2 = pd.DataFrame(index=d.index)
    d_target2['context'] = ['C2']*len(d_target2)

    # how much S1 is suppressed in C1 (where S1 punished)
    d_target1['score'] = d[('baseline', 'C1')] - d[('training', 'C1')]
    # how much S1 is enhanced in C2 (where S2 punished)
    d_target2['score'] = - (d[('baseline', 'C2')] - d[('training', 'C2')])

    # what proportion of baseline S1 is suppressed in C1 (where S1 punished)
    d_target1['relative_score'] =  1 - (d[('training', 'C1')] / d[('baseline', 'C1')])
    # what proportion of baseline S1 is enhanced in C2 (where S2 punished)
    d_target2['relative_score'] = 1 - ((1-d[('training', 'C2')]) / (1-d[('baseline', 'C2')]))

    # post-training difference score
    d_target1['posttrain_difference_score'] =  d[('training', 'C2')] - d[('training', 'C1')]
    # what proportion of baseline S1 is enhanced in C2 (where S2 punished)
    d_target2['posttrain_difference_score'] = d[('training', 'C2')] - d[('training', 'C1')]

    # likelihood of S1 when S1 is suppressed
    d_target1['p_syllable_neg'] = [mean_likelihoods[subject][0] for subject in d.index]
    # likelihood of S2 when S1 is suppressed
    d_target1['p_syllable_pos'] = [mean_likelihoods[subject][1] for subject in d.index]
    # likelihood of S1 when S2 is suppressed
    d_target2['p_syllable_pos'] = [mean_likelihoods[subject][0] for subject in d.index]
    # likelihood of S1 when S2 is suppressed
    d_target2['p_syllable_neg'] = [mean_likelihoods[subject][1] for subject in d.index]

    # bigram frequency of S1 when S1 is suppressed
    d_target1['p_bigram_neg'] = [bigram_probabilities[subject][0] for subject in d.index]
    # bigram frequency of S2 when S1 is suppressed
    d_target1['p_bigram_pos'] = [bigram_probabilities[subject][1] for subject in d.index]
    # bigram frequency of S1 when S2 is suppressed
    d_target2['p_bigram_pos'] = [bigram_probabilities[subject][0] for subject in d.index]
    # bigram frequency of S1 when S2 is suppressed
    d_target2['p_bigram_neg'] = [bigram_probabilities[subject][1] for subject in d.index]

    # depth of S1 when S1 is suppressed
    d_target1['d_syllable_neg'] = [median_context_depths[subject][0] for subject in d.index]
    # depth of S2 when S1 is suppressed
    d_target1['d_syllable_pos'] = [median_context_depths[subject][1] for subject in d.index]
    # depth of S1 when S2 is suppressed
    d_target2['d_syllable_pos'] = [median_context_depths[subject][0] for subject in d.index]
    # depth of S1 when S2 is suppressed
    d_target2['d_syllable_neg'] = [median_context_depths[subject][1] for subject in d.index]

    d = pd.concat([d_target1, d_target2])
    d['score']                      = d['score'].astype('float') * 100
    d['relative_score']             = d['relative_score'].astype('float') * 100
    d['posttrain_difference_score'] = d['posttrain_difference_score'].astype('float') * 100

    return d
