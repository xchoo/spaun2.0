import numpy as np
import nengo
from nengo import spa

from .configurator import cfg
from .vocabulator import vocab
from .loggerator import logger
from _spaun.modules import Stimulus, Vision, ProdSys, RewardEval, InfoEnc
from _spaun.modules import TrfmSys, Memory, Monitor, InfoDec, Motor
from _spaun.modules import InstrStimulus, InstrProcess

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
        model.instr_stim = InstrStimulus()
        model.vis = Vision()
        model.ps = ProdSys()
        model.reward = RewardEval()
        model.enc = InfoEnc()
        model.mem = Memory()
        model.trfm = TrfmSys()
        model.dec = InfoDec()
        model.mtr = Motor()
        model.instr = InstrProcess()
        model.monitor = Monitor()

        model.learn_conns = []

        if hasattr(model, 'vis') and hasattr(model, 'ps'):
            copy_draw_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, ZER)) --> ps_task = W, ps_state = TRANS0, ps_dec = FWD',  # noqa
                 'dot(ps_task, W-DEC) - dot(vis, QM) --> ps_state = ps_state']  # noqa
            recog_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, ONE)) --> ps_task = R, ps_state = TRANS0, ps_dec = FWD',  # noqa
                 'dot(ps_task, R-DEC) - dot(vis, QM) --> ps_state = ps_state']  # noqa

            learn_state_action = ['0.5 * (dot(ps_task, X) + dot(vis, TWO)) - dot(vis, QM) --> ps_task = L, ps_state = LEARN, ps_dec = FWD']  # noqa
                                  # '0.5 * (dot(ps_task, L) + dot(vis, {:s} + {:s})) --> ps_state = LEARN'.format(*vocab.reward_sp_strs)]  # noqa
            if hasattr(model, 'reward'):
                learn_action = ['0.5 * dot(ps_task, L) - dot(vis, QM) --> ps_action = %s, ps_state = LEARN, ps_dec = NONE' %  # noqa
                                s for s in vocab.ps_action_learn_sp_strs]
            else:
                learn_action = []

            mem_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, THR)) --> ps_task = M, ps_state = TRANS0, ps_dec = FWD',  # noqa
                 'dot(ps_task, M-DEC) - dot(vis, F + R + QM) --> ps_state = ps_state',  # noqa
                 '0.5 * (dot(ps_task, M-DEC) + dot(vis, F)) - dot(vis, QM) --> ps_dec = FWD',  # noqa
                 '0.5 * (dot(ps_task, M-DEC) + dot(vis, R)) - dot(vis, QM) --> ps_dec = REV']  # noqa

            if hasattr(model, 'trfm'):
                count_action = \
                    ['0.5 * (dot(ps_task, X) + dot(vis, FOR)) --> ps_task = C, ps_state = TRANS0, ps_dec = FWD',  # noqa
                     '0.5 * (dot(ps_task, C-DEC) + dot(ps_state, TRANS0)) - dot(vis, QM) --> ps_state = CNT0',  # noqa
                     '0.5 * (dot(ps_task, C-DEC) + dot(ps_state, CNT0)) - dot(vis, QM) --> ps_state = CNT1',  # noqa
                     '(0.25 * (dot(ps_task, DEC) + dot(ps_state, CNT1)) + 0.5 * dot(trfm_compare, NO_MATCH)) + (dot(ps_dec, CNT) - 1) - dot(vis, QM) --> ps_dec = CNT, ps_state = CNT1',  # noqa
                     '(0.25 * (dot(ps_task, DEC) + dot(ps_state, CNT1)) + 0.5 * dot(trfm_compare, MATCH)) + (dot(ps_dec, CNT) - 1) - dot(vis, QM) --> ps_dec = FWD, ps_state = TRANS0']  # noqa
                qa_action = \
                    ['0.5 * (dot(ps_task, X) + dot(vis, FIV)) --> ps_task = A, ps_state = TRANS0, ps_dec = FWD',  # noqa
                     'dot(ps_task, A-DEC) - dot(vis, K + P + QM) --> ps_state = ps_state',  # noqa
                     '0.5 * (dot(ps_task, A-DEC) + dot(vis, K)) - dot(vis, QM) --> ps_state = QAK',  # noqa
                     '0.5 * (dot(ps_task, A-DEC) + dot(vis, P)) - dot(vis, QM) --> ps_state = QAP']  # noqa
                rvc_action = \
                    ['0.5 * (dot(ps_task, X) + dot(vis, SIX)) --> ps_task = V, ps_state = TRANS0, ps_dec = FWD',  # noqa
                     '0.5 * (dot(ps_task, V-DEC) + dot(ps_state, TRANS0)) - dot(vis, QM) --> ps_state = TRANS1',  # noqa
                     '0.5 * (dot(ps_task, V-DEC) + dot(ps_state, TRANS1)) - dot(vis, QM) --> ps_state = TRANS0']  # noqa
                fi_action = \
                    ['0.5 * (dot(ps_task, X) + dot(vis, SEV)) --> ps_task = F, ps_state = TRANS0, ps_dec = FWD',  # noqa
                     '0.5 * (dot(ps_task, F-DEC) + dot(ps_state, TRANS0)) - dot(vis, QM) --> ps_state = TRANS1',  # noqa
                     '0.5 * (dot(ps_task, F-DEC) + dot(ps_state, TRANS1)) - dot(vis, QM) --> ps_state = TRANS2',  # noqa
                     '0.5 * (dot(ps_task, F-DEC) + dot(ps_state, TRANS2)) - dot(vis, QM) --> ps_state = TRANS0']  # noqa

                # Reaction task
                react_action = \
                    ['0.5 * (dot(ps_task, X) + dot(vis, EIG)) --> ps_task = REACT, ps_state = DIRECT, ps_dec = FWD',  # noqa
                     '0.5 * (dot(ps_task, REACT) + dot(vis_mem, ONE)) --> trfm_input = POS1*THR',  # noqa
                     '0.5 * (dot(ps_task, REACT) + dot(vis_mem, TWO)) --> trfm_input = POS1*FOR']  # noqa
            else:
                count_action = []
                qa_action = []
                rvc_action = []
                fi_action = []
                react_action = []

            if hasattr(model, 'trfm') and hasattr(model, 'instr'):
                instr_action = \
                    ['0.5 * (dot(ps_task, X) + dot(vis, NIN)) --> ps_task = INSTR, ps_state = DIRECT, ps_dec = FWD',  # noqa
                     # 'dot(ps_task, INSTR) - dot(vis, QM + A) --> trfm_input = instr_data',  # noqa - Note: this is the bg 'instr' rule in it's simplified form
                     'dot(ps_task, INSTR) - dot(vis, QM + A + M + P + CLOSE) - dot(ps_state, INSTRP) --> instr_en = ENABLE, ps_task = instr_task, ps_state = instr_state, ps_dec = instr_dec, trfm_input = instr_data',  # noqa
                     '1.2 * dot(vis, M) --> ps_task = INSTR, ps_state = TRANS0, ps_dec = FWD',   # noqa
                     '0.5 * (dot(ps_task, INSTR) + dot(vis, P)) --> ps_task = INSTR, ps_state = INSTRP',   # noqa
                     '1.2 * dot(vis, V) --> ps_task = INSTR, ps_state = INSTRV',   # noqa
                     '0.5 * (dot(ps_task, INSTR) + dot(ps_state, INSTRP)) --> ps_task = INSTR, ps_state = TRANS0',   # noqa
                     # 'dot(instr_util, INSTR) - dot(ps_task, INSTR) --> instr_en = ENABLE, ps_task = instr_task, ps_state = instr_state, ps_dec = instr_dec, trfm_input = instr_data',  # noqa
                     ]  # noqa
            else:
                instr_action = []

            decode_action = \
                ['dot(vis, QM) - 0.75 * dot(ps_task, W + C + V + F + L + REACT) --> ps_task = ps_task + DEC, ps_state = ps_state + 0.6 * TRANS0, ps_dec = ps_dec + 0.6 * FWD',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, W)) --> ps_task = W + DEC, ps_state = ps_state, ps_dec = DECW',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, C)) --> ps_task = C + DEC, ps_state = ps_state, ps_dec = CNT',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, V + F)) --> ps_task = ps_task + DEC, ps_state = ps_state, ps_dec = DECI',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, L)) --> ps_task = L + DEC, ps_state = LEARN, ps_dec = FWD',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, REACT)) --> ps_task = REACT + DEC, ps_state = DIRECT, ps_dec = FWD',  # noqa
                 '0.75 * dot(ps_task, DEC-REACT-INSTR) - dot(ps_state, LEARN) - dot(ps_dec, CNT) - dot(vis, QM + A + M) --> ps_task = ps_task, ps_state = ps_state, ps_dec = ps_dec']  # noqa
            default_action = \
                []
                # ['0.80 * dot(vis, A) --> ps_task = X']  # noqa

            # List learning task spa actions first, so we know the precise
            # indicies of the learning task actions (i.e. the first N)
            all_actions = (learn_action + learn_state_action +
                           copy_draw_action + recog_action + mem_action +
                           count_action + qa_action + rvc_action + fi_action +
                           decode_action + default_action + react_action +
                           instr_action)

            actions = spa.Actions(*all_actions)
            model.bg = spa.BasalGanglia(actions=actions, input_synapse=0.008,
                                        label='Basal Ganglia')
            model.thal = spa.Thalamus(model.bg, subdim_channel=1,
                                      mutual_inhibit=1, route_inhibit=5.0,
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

                    for i in range(len(learn_action)):
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
            # Modify any 'channel' ensemble arrays to have
            # get_optimal_sp_radius radius sizes
            for net in model.trfm.all_networks:
                if net.label is not None and net.label[:7] == 'channel':
                    for ens in net.all_ensembles:
                        ens.radius = cfg.get_optimal_sp_radius()
        if hasattr(model, 'dec'):
            model.dec.setup_connections(model)
        if hasattr(model, 'mtr'):
            model.mtr.setup_connections(model)
        if hasattr(model, 'instr'):
            model.instr.setup_connections(model)
        if hasattr(model, 'monitor'):
            model.monitor.setup_connections(model)

    return model
