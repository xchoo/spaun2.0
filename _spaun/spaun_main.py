import numpy as np
import nengo
from nengo import spa

from .configurator import cfg
from .vocabulator import vocab
from .experimenter import experiment
from .loggerator import logger
from _spaun.modules import Stimulus, Vision, ProdSys, RewardEval, InfoEnc
from _spaun.modules import TrfmSys, Memory, Monitor, InfoDec, Motor

# #### DEBUG DUMMY NETWORK IMPORTS ####
# from _spaun.modules.experimenter import StimulusDummy as Stimulus  # noqa
# from _spaun.modules.vision_system import VisionSystemDummy as Vision  # noqa
# from _spaun.modules.working_memory import WorkingMemoryDummy as Memory  # noqa
# from _spaun.modules.transform_system import TransformationSystemDummy as TrfmSys  # noqa


def Spaun():
    model = spa.SPA(label='Spaun', seed=cfg.seed)
    with model:
        model.config[nengo.Ensemble].max_rates = cfg.max_rates
        model.config[nengo.Ensemble].neuron_type = cfg.neuron_type
        model.config[nengo.Ensemble].n_neurons = cfg.n_neurons_ens
        model.config[nengo.Connection].synapse = cfg.pstc

        model.stim = Stimulus()
        model.vis = Vision()
        model.ps = ProdSys()
        model.reward = RewardEval()
        model.enc = InfoEnc()
        model.mem = Memory()
        model.trfm = TrfmSys()
        model.dec = InfoDec()
        model.mtr = Motor()
        model.monitor = Monitor()

        model.learn_conns = []

        if hasattr(model, 'vis') and hasattr(model, 'ps'):
            copy_draw_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, ZER)) --> ps_task = W',  # noqa
                 'dot(ps_task, W-DEC) - dot(vis, QM) --> ps_state = ps_state']  # noqa
            recog_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, ONE)) --> ps_task = R',  # noqa
                 'dot(ps_task, R-DEC) - dot(vis, QM) --> ps_state = ps_state']  # noqa

            learn_state_action = ['0.5 * (dot(ps_task, X) + dot(vis, TWO)) - dot(vis, QM) --> ps_task = L']  # noqa
                                  # '0.5 * (dot(ps_task, L) + dot(vis, {:s} + {:s})) --> ps_state = LEARN'.format(*vocab.reward_sp_strs)]  # noqa
            if hasattr(model, 'reward'):
                learn_action = ['0.5 * dot(ps_task, L) - dot(vis, QM) --> ps_action = %s, ps_state = LEARN, ps_dec = NONE' %  # noqa
                                s for s in vocab.ps_action_learn_sp_strs]
            else:
                learn_action = []

            mem_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, THR)) --> ps_task = M',  # noqa
                 'dot(ps_task, M-DEC) - dot(vis, F + R + QM) --> ps_state = ps_state',  # noqa
                 '0.5 * (dot(ps_task, M-DEC) + dot(vis, F)) - dot(vis, QM) --> ps_dec = FWD',  # noqa
                 '0.5 * (dot(ps_task, M-DEC) + dot(vis, R)) - dot(vis, QM) --> ps_dec = REV']  # noqa

            if hasattr(model, 'trfm'):
                count_action = \
                    ['0.5 * (dot(ps_task, X) + dot(vis, FOR)) --> ps_task = C',  # noqa
                     '0.5 * (dot(ps_task, C-DEC) + dot(ps_state, TRANS0)) - dot(vis, QM) --> ps_state = CNT0',  # noqa
                     '0.5 * (dot(ps_task, C-DEC) + dot(ps_state, CNT0)) - dot(vis, QM) --> ps_state = CNT1',  # noqa
                     '(0.25 * (dot(ps_task, DEC) + dot(ps_state, CNT1)) + 0.5 * dot(trfm_compare, NO_MATCH)) + (dot(ps_dec, CNT) - 1) - dot(vis, QM) --> ps_dec = CNT, ps_state = CNT1',  # noqa
                     '(0.25 * (dot(ps_task, DEC) + dot(ps_state, CNT1)) + 0.5 * dot(trfm_compare, MATCH)) + (dot(ps_dec, CNT) - 1) - dot(vis, QM) --> ps_dec = FWD, ps_state = TRANS0']  # noqa
                qa_action = \
                    ['0.5 * (dot(ps_task, X) + dot(vis, FIV)) --> ps_task = A',  # noqa
                     'dot(ps_task, A-DEC) - dot(vis, K + P + QM) --> ps_state = ps_state',  # noqa
                     '0.5 * (dot(ps_task, A-DEC) + dot(vis, K)) - dot(vis, QM) --> ps_state = QAK',  # noqa
                     '0.5 * (dot(ps_task, A-DEC) + dot(vis, P)) - dot(vis, QM) --> ps_state = QAP']  # noqa
                rvc_action = \
                    ['0.5 * (dot(ps_task, X) + dot(vis, SIX)) --> ps_task = V',  # noqa
                     '0.5 * (dot(ps_task, V-DEC) + dot(ps_state, TRANS0)) - dot(vis, QM) --> ps_state = TRANS1',  # noqa
                     '0.5 * (dot(ps_task, V-DEC) + dot(ps_state, TRANS1)) - dot(vis, QM) --> ps_state = TRANS0']  # noqa
                fi_action = \
                    ['0.5 * (dot(ps_task, X) + dot(vis, SEV)) --> ps_task = F',  # noqa
                     '0.5 * (dot(ps_task, F-DEC) + dot(ps_state, TRANS0)) - dot(vis, QM) --> ps_state = TRANS1',  # noqa
                     '0.5 * (dot(ps_task, F-DEC) + dot(ps_state, TRANS1)) - dot(vis, QM) --> ps_state = TRANS2',  # noqa
                     '0.5 * (dot(ps_task, F-DEC) + dot(ps_state, TRANS2)) - dot(vis, QM) --> ps_state = TRANS0']  # noqa
            else:
                count_action = []
                qa_action = []
                rvc_action = []
                fi_action = []

            decode_action = \
                ['dot(vis, QM) - 0.75 * dot(ps_task, W + C + V + F + L) --> ps_task = DEC, ps_state = ps_state, ps_dec = ps_dec',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, W)) --> ps_task = DEC, ps_state = ps_state, ps_dec = DECW',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, C)) --> ps_task = DEC, ps_state = ps_state, ps_dec = CNT',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, V + F)) --> ps_task = DEC, ps_state = ps_state, ps_dec = DECI',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, L)) --> ps_task = L, ps_state = LEARN, ps_dec = FWD',  # noqa
                 'dot(ps_task, DEC) - dot(ps_dec, CNT) - dot(vis, QM) --> ps_task = ps_task, ps_state = ps_state, ps_dec = ps_dec']  # noqa
            default_action = \
                []

            # List learning task spa actions first, so we know the precise
            # indicies of the learning task actions (i.e. the first N)
            all_actions = (learn_action + learn_state_action +
                           copy_draw_action + recog_action + mem_action +
                           count_action + qa_action + rvc_action + fi_action +
                           decode_action + default_action)

            actions = spa.Actions(*all_actions)
            model.bg = spa.BasalGanglia(actions=actions, input_synapse=0.008,
                                        label='Basal Ganglia')
            model.thal = spa.Thalamus(model.bg, mutual_inhibit=1,
                                      label='Thalamus')

        # ----- Set up connections (and save record of modules) -----
        if hasattr(model, 'vis'):
            model.vis.setup_connections(model)
        if hasattr(model, 'ps'):
            model.ps.setup_connections(model)
        if hasattr(model, 'bg'):
            if hasattr(model, 'reward'):
                with model.bg:
                    # Generate random biases for each learn action, so that
                    # there is some randomness to the initial action choice
                    bias_node = nengo.Node(1)
                    bias_ens = nengo.Ensemble(cfg.n_neurons_ens, 1,
                                              label='BG Bias Ensemble')
                    nengo.Connection(bias_node, bias_ens)

                    for i in range(experiment.num_learn_actions):
                        init_trfm = (np.random.random() *
                                     cfg.learn_init_trfm_max)
                        trfm_val = cfg.learn_init_trfm_bias + init_trfm
                        model.learn_conns.append(
                            nengo.Connection(bias_ens, model.bg.input[i],
                                             transform=trfm_val))
                        cfg.learn_init_transforms.append(trfm_val)
                logger.write("# learn_init_trfms: %s\n" %
                             (str(cfg.learn_init_transforms)))
        if hasattr(model, 'thal'):
            pass
        if hasattr(model, 'reward'):
            model.reward.setup_connections(model, model.learn_conns)
        if hasattr(model, 'enc'):
            model.enc.setup_connections(model)
        if hasattr(model, 'mem'):
            model.mem.setup_connections(model)
        if hasattr(model, 'trfm'):
            model.trfm.setup_connections(model)
        if hasattr(model, 'dec'):
            model.dec.setup_connections(model)
        if hasattr(model, 'mtr'):
            model.mtr.setup_connections(model)
        if hasattr(model, 'monitor'):
            model.monitor.setup_connections(model)

    return model
