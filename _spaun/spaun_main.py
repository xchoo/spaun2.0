import numpy as np
import nengo
from nengo import spa

from .configurator import cfg
from .vocabulator import vocab
from .loggerator import logger
from .modules import Stimulus, Vision, ExecSys, RewardEval, InfoEnc
from .modules import TrfmSys, Memory, Monitor, InfoDec, Motor
from .modules import InstrStimulus, InstrProcess

# #### DEBUG DUMMY NETWORK IMPORTS ####
# from _spaun.modules.experimenter import StimulusDummy as Stimulus  # noqa
# from _spaun.modules.vision_system import VisionSystemDummy as Vision  # noqa
# from _spaun.modules.working_memory import WorkingMemoryDummy as Memory  # noqa
# from _spaun.modules.transform_system import TransformationSystemDummy as TrfmSys  # noqa
# from _spaun.modules.motor_system import MotorSystemDummy as Motor


def Spaun():
    model = spa.SPA(label='Spaun', seed=cfg.seed)
    with model:
        model.config[nengo.Ensemble].max_rates = cfg.max_rates
        model.config[nengo.Ensemble].neuron_type = cfg.neuron_type
        model.config[nengo.Connection].synapse = cfg.pstc

        if 'S' in cfg.spaun_modules:
            model.stim = Stimulus()
            model.instr_stim = InstrStimulus()
            model.monitor = Monitor()
        if 'V' in cfg.spaun_modules:
            model.vis = Vision()
        if 'X' in cfg.spaun_modules:
            model.exe = ExecSys()
        if 'R' in cfg.spaun_modules:
            model.reward = RewardEval()
        if 'E' in cfg.spaun_modules:
            model.enc = InfoEnc()
        if 'W' in cfg.spaun_modules:
            model.mem = Memory()
        if 'T' in cfg.spaun_modules:
            model.trfm = TrfmSys()
        if 'D' in cfg.spaun_modules:
            model.dec = InfoDec()
        if 'M' in cfg.spaun_modules:
            model.mtr = Motor()
        if 'I' in cfg.spaun_modules:
            model.instr = InstrProcess()

        model.learn_conns = []

        if hasattr(model, 'vis') and hasattr(model, 'exe'):
            copy_draw_action = \
                ['0.5 * (dot(exe_task, X) + dot(vis, ZER)) --> exe_task = W, exe_state = TRANS0, exe_dec = FWD',  # noqa
                 'dot(exe_task, W-DEC) - dot(vis, QM) --> exe_state = exe_state']  # noqa
            # Copy drawing task format: A0[r]?X

            recog_action = \
                ['0.5 * (dot(exe_task, X) + dot(vis, ONE)) --> exe_task = R, exe_state = TRANS0, exe_dec = FWD',  # noqa
                 'dot(exe_task, R-DEC) - dot(vis, QM) --> exe_state = exe_state']  # noqa
            # Digit recognition task format: A1[r]?X

            learn_state_action = ['0.5 * (dot(exe_task, X) + dot(vis, TWO)) - dot(vis, QM) --> exe_task = L, exe_state = LEARN, exe_dec = FWD']  # noqa
            # Learning task format: A2?X<REWARD>?X<REWARD>?X<REWARD>?X<REWARD>?X<REWARD>    # noqa
            if hasattr(model, 'reward'):
                learn_action = ['0.5 * (dot(exe_task, 2*L) - 1) - dot(vis, QM) --> exe_action = %s, exe_state = LEARN, exe_dec = NONE' %  # noqa
                                s for s in vocab.exe_action_learn_sp_strs]
            else:
                learn_action = []

            mem_action = \
                ['0.5 * (dot(exe_task, X) + dot(vis, THR)) --> exe_task = M, exe_state = TRANS0, exe_dec = FWD',  # noqa
                 'dot(exe_task, M-DEC) - dot(vis, F + R + QM) --> exe_state = exe_state',  # noqa
                 '0.5 * (dot(exe_task, M) + dot(vis, F)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_dec = FWD',  # noqa
                 '0.5 * (dot(exe_task, M) + dot(vis, R)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_dec = REV']  # noqa
            # Working memory task format: A3[rr..rr]?XXX
            # Reverse recall task format: A3[rr..rr]R?XXX

            if hasattr(model, 'trfm'):
                count_action = \
                    ['0.5 * (dot(exe_task, X) + dot(vis, FOR)) --> exe_task = C, exe_state = TRANS0, exe_dec = FWD',  # noqa
                     '0.5 * (dot(exe_task, C) + dot(exe_state, TRANS0)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_state = CNT0',  # noqa
                     '0.5 * (dot(exe_task, C) + dot(exe_state, CNT0)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_state = CNT1',  # noqa
                     '(0.25 * (dot(exe_task, DEC) + dot(exe_state, CNT1)) + 0.5 * dot(trfm_compare, NO_MATCH)) + (dot(exe_dec, CNT) - 1) - dot(vis, QM) --> exe_dec = CNT, exe_state = CNT1',  # noqa
                     '(0.25 * (dot(exe_task, DEC) + dot(exe_state, CNT1)) + 0.5 * dot(trfm_compare, MATCH)) + (dot(exe_dec, CNT) - 1) - dot(vis, QM) --> exe_dec = FWD, exe_state = TRANS0']  # noqa
                # Counting task format: A4[START_NUM][NUM_COUNT]?X..X
                qa_action = \
                    ['0.5 * (dot(exe_task, X) + dot(vis, FIV)) --> exe_task = A, exe_state = TRANS0, exe_dec = FWD',  # noqa
                     'dot(exe_task, A-DEC) - dot(vis, K + P + QM) --> exe_state = exe_state',  # noqa
                     '0.5 * (dot(exe_task, A) + dot(vis, K)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_state = QAK',  # noqa
                     '0.5 * (dot(exe_task, A) + dot(vis, P)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_state = QAP']  # noqa
                # Question answering task format: A5[rr..rr]P[r]?X (probing item in position)           # noqa
                #                                 A5[rr..rr]K[r]?X (probing position of item (kind))    # noqa
                rvc_action = \
                    ['0.5 * (dot(exe_task, X) + dot(vis, SIX)) --> exe_task = V, exe_state = TRANS0, exe_dec = FWD',  # noqa
                     '0.5 * (dot(exe_task, V) + dot(exe_state, TRANS0)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_state = TRANS1',  # noqa
                     '0.5 * (dot(exe_task, V) + dot(exe_state, TRANS1)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_state = TRANS0']  # noqa
                # Rapid variable creation task format: A6{[rr..rr][rr..rr]:NUM_EXAMPLES}?XX..XX     # noqa
                fi_action = \
                    ['0.5 * (dot(exe_task, X) + dot(vis, SEV)) --> exe_task = F, exe_state = TRANS0, exe_dec = FWD',  # noqa
                     '0.5 * (dot(exe_task, F) + dot(exe_state, TRANS0)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_state = TRANS1',  # noqa
                     '0.5 * (dot(exe_task, F) + dot(exe_state, TRANS1)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_state = TRANS2',  # noqa
                     '0.5 * (dot(exe_task, F) + dot(exe_state, TRANS2)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_state = TRANS0']  # noqa
                # Fluid intelligence task format: A7[CELL1_1][CELL1_2][CELL1_3][CELL2_1][CELL2_2][CELL2_3][CELL3_1][CELL3_2]?XX..XX     # noqa

                # Reaction task
                react_action = \
                    ['0.5 * (dot(exe_task, X) + dot(vis, EIG)) --> exe_task = REACT, exe_state = DIRECT, exe_dec = FWD',  # noqa
                     '0.5 * (dot(exe_task, REACT) + dot(vis_mem, ONE)) --> trfm_input = POS1*THR',  # noqa
                     '0.5 * (dot(exe_task, REACT) + dot(vis_mem, TWO)) --> trfm_input = POS1*FOR']  # noqa
                # Stimulus response (hardcoded reaction) task format: A8?1X<expected 3>?2X<expected 4>    # noqa

                # Compare task -- See two items, check if their class matches each other # noqa
                match_action = \
                    ['0.5 * (dot(exe_task, X) + dot(vis, C)) --> exe_task = CMP, exe_state = TRANS1, exe_dec = FWD',  # noqa
                     '0.5 * (dot(exe_task, CMP) + dot(exe_state, TRANS1)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_state = TRANS2',  # noqa
                     '0.5 * (dot(exe_task, CMP) + dot(exe_state, TRANS2)) - dot(vis, QM) - dot(exe_task, DEC) --> exe_state = TRANSC']  # noqa
                # List / item matching task format: AC[r][r]?X<expected 1 if match, 0 if not>                                                      # noqa
                #                                   AC[rr..rr][rr..rr]?X<expected 1 if any item in first list appears in second list, 0 if not>    # noqa
            else:
                count_action = []
                qa_action = []
                rvc_action = []
                fi_action = []
                react_action = []
                match_action = []

            if hasattr(model, 'trfm') and hasattr(model, 'instr'):
                instr_action = \
                    ['0.5 * (dot(exe_task, X) + dot(vis, NIN)) --> exe_task = INSTR, exe_state = DIRECT, exe_dec = FWD',  # noqa
                     # 'dot(exe_task, INSTR) - dot(vis, QM + A) --> trfm_input = instr_data',  # noqa - Note: this is the bg 'instr' rule in it's simplified form
                     'dot(exe_task, INSTR) - dot(vis, QM + A + M + P + CLOSE) - dot(exe_state, INSTRP) --> instr_en = ENABLE, exe_task = instr_task, exe_state = instr_state, exe_dec = instr_dec, trfm_input = instr_data',  # noqa
                     '1.5 * dot(vis, M + V) --> exe_task = INSTR, exe_state = TRANS0, exe_dec = FWD',   # noqa - Note: V no longer keeps state, dec information from before. Need to set in instruction
                     '0.5 * (dot(exe_task, INSTR) + dot(vis, P)) --> exe_task = INSTR, exe_state = INSTRP',   # noqa
                     '0.5 * (dot(exe_task, INSTR) + dot(exe_state, INSTRP)) --> exe_task = INSTR, exe_state = TRANS0',   # noqa
                     # 'dot(instr_util, INSTR) - dot(exe_task, INSTR) --> instr_en = ENABLE, exe_task = instr_task, exe_state = instr_state, exe_dec = instr_dec, trfm_input = instr_data',  # noqa - Note: this is the untested 'use output of IPS to set PS states' implementation
                     ]  # noqa
                # Instructed tasks task formats:
                # Instructed stimulus response task format: A9?rX<expected answer from instruction>?rX<expected answer from instruction>    # noqa
                # Instructed custom task format: M<0-9>[INSTRUCTED TASK FORMAT]?XX..XX                                                      # noqa
                # Instructed positional task formats: MP<0-9>[INSTRUCTED TASK FORMAT]?XX..XX V[INSTRUCTED TASK FORMAT]?XX..XX               # noqa
                #     - P<0-9> selects appropriate instruction from list of instructions                                                    # noqa
                #     - V increments instruction position by 1
            else:
                instr_action = []

            decode_action = \
                ['dot(vis, QM) - 0.6 * dot(exe_task, W+C+V+F+L+REACT) --> exe_task = exe_task + DEC, exe_state = exe_state + 0.5 * TRANS0, exe_dec = exe_dec + 0.5 * FWD',  # noqa
                 '0.5 * (dot(vis, QM) + dot(exe_task, W-DEC)) --> exe_task = W + DEC, exe_state = exe_state, exe_dec = DECW',  # noqa
                 '0.5 * (dot(vis, QM) + dot(exe_task, C-DEC)) --> exe_task = C + DEC, exe_state = exe_state, exe_dec = CNT',  # noqa
                 '0.5 * (dot(vis, QM) + dot(exe_task, V+F-DEC)) --> exe_task = exe_task + DEC, exe_state = exe_state, exe_dec = DECI',  # noqa
                 '0.7 * dot(vis, QM) + 0.3 * dot(exe_task, L) --> exe_task = L + DEC, exe_state = LEARN, exe_dec = FWD',  # noqa
                 '0.5 * (dot(vis, QM) + dot(exe_task, REACT)) --> exe_task = REACT + DEC, exe_state = DIRECT, exe_dec = FWD',  # noqa
                 '0.75 * dot(exe_task, DEC-REACT-INSTR) - dot(exe_state, LEARN) - dot(exe_dec, CNT) - dot(vis, QM + A + M) --> exe_task = exe_task, exe_state = exe_state, exe_dec = exe_dec']  # noqa
            default_action = \
                []

            # List learning task spa actions first, so we know the precise
            # indicies of the learning task actions (i.e. the first N)
            all_actions = (learn_action + learn_state_action +
                           copy_draw_action + recog_action + mem_action +
                           count_action + qa_action + rvc_action + fi_action +
                           decode_action + default_action + react_action +
                           instr_action + match_action)

            actions = spa.Actions(*all_actions)
            model.bg = spa.BasalGanglia(actions=actions, input_synapse=0.008,
                                        label='Basal Ganglia')
            model.thal = spa.Thalamus(model.bg, subdim_channel=1,
                                      mutual_inhibit=1, route_inhibit=5.0,
                                      label='Thalamus')

        # ----- Set up connections (and save record of modules) -----
        if hasattr(model, 'vis'):
            model.vis.setup_connections(model)
        if hasattr(model, 'exe'):
            model.exe.setup_connections(model)
            # Modify any 'channel' ensemble arrays to have
            # get_optimal_sp_radius radius sizes
            for net in model.exe.all_networks:
                if net.label is not None and net.label[:7] == 'channel':
                    for ens in net.all_ensembles:
                        ens.radius = cfg.get_optimal_sp_radius()
        if hasattr(model, 'bg'):
            if hasattr(model, 'reward'):
                # Clear learning transforms
                del cfg.learn_init_transforms[:]

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
