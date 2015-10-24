import nengo
from nengo import spa

from .config import cfg
from _spaun.modules.stimulus import parse_raw_seq
from _spaun.modules import Stimulus, Vision, ProdSys, InfoEnc, InfoDec, Motor
from _spaun.modules import TrfmSys, Memory

# #### DEBUG DUMMY NETWORK IMPORTS ####
# from _spaun.modules.stimulus import StimulusDummy as Stimulus  # noqa
# from _spaun.modules.vision_system import VisionSystemDummy as Vision  # noqa
# from _spaun.modules.working_memory import WorkingMemoryDummy as Memory  # noqa
# from _spaun.modules.transform_system import TransformationSystemDummy as TrfmSys  # noqa


def Spaun():
    # Process the raw stimulus provided to spaun
    parse_raw_seq()

    model = spa.SPA(label='Spaun', seed=cfg.seed)
    with model:
        model.config[nengo.Ensemble].max_rates = cfg.max_rates
        model.config[nengo.Ensemble].neuron_type = cfg.neuron_type
        model.config[nengo.Ensemble].n_neurons = cfg.n_neurons_ens
        model.config[nengo.Connection].synapse = cfg.pstc

        model.stim = Stimulus()
        model.vis = Vision()
        model.ps = ProdSys()
        model.enc = InfoEnc()
        model.mem = Memory()
        model.trfm = TrfmSys()
        model.dec = InfoDec()
        model.mtr = Motor()

        if hasattr(model, 'vis') and hasattr(model, 'ps') and \
           hasattr(model, 'trfm'):
            copy_draw_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, ZER)) --> ps_task = W',  # noqa
                 'dot(ps_task, W-DEC) - dot(vis, QM) --> ps_task = W, ps_state = ps_state']  # noqa
            recog_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, ONE)) --> ps_task = R',  # noqa
                 'dot(ps_task, R-DEC) - dot(vis, QM) --> ps_task = R, ps_state = ps_state']  # noqa
            mem_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, THR)) --> ps_task = M',  # noqa
                 'dot(ps_task, M-DEC) - dot(vis, F + R + QM) --> ps_task = M, ps_state = ps_state',  # noqa
                 '0.5 * (dot(ps_task, M-DEC) + dot(vis, F)) - dot(vis, QM) --> ps_task = M, ps_dec = FWD',  # noqa
                 '0.5 * (dot(ps_task, M-DEC) + dot(vis, R)) - dot(vis, QM) --> ps_task = M, ps_dec = REV']  # noqa
            count_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, FOR)) --> ps_task = C',  # noqa
                 '0.5 * (dot(ps_task, C-DEC) + dot(ps_state, TRANS0)) - dot(vis, QM) --> ps_task = C, ps_state = CNT0',  # noqa
                 '0.5 * (dot(ps_task, C-DEC) + dot(ps_state, CNT0)) - dot(vis, QM) --> ps_task = C, ps_state = CNT1',  # noqa
                 '(0.25 * (dot(ps_task, DEC) + dot(ps_state, CNT1)) + 0.5 * dot(trfm_compare, NO_MATCH)) + (dot(ps_dec, CNT) - 1) - dot(vis, QM) --> ps_dec = CNT, ps_state = CNT1',  # noqa
                 '(0.25 * (dot(ps_task, DEC) + dot(ps_state, CNT1)) + 0.5 * dot(trfm_compare, MATCH)) + (dot(ps_dec, CNT) - 1) - dot(vis, QM) --> ps_dec = FWD, ps_state = TRANS0']  # noqa
            qa_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, FIV)) --> ps_task = A',  # noqa
                 'dot(ps_task, A-DEC) - dot(vis, K + P + QM) --> ps_task = A, ps_state = ps_state',  # noqa
                 '0.5 * (dot(ps_task, A-DEC) + dot(vis, K)) - dot(vis, QM) --> ps_task = M, ps_state = QAK',  # noqa
                 '0.5 * (dot(ps_task, A-DEC) + dot(vis, P)) - dot(vis, QM) --> ps_task = M, ps_state = QAP']  # noqa
            rvc_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, SIX)) --> ps_task = V',  # noqa
                 '0.5 * (dot(ps_task, V-DEC) + dot(ps_state, TRANS0)) - dot(vis, QM) --> ps_task = V, ps_state = TRANS1',  # noqa
                 '0.5 * (dot(ps_task, V-DEC) + dot(ps_state, TRANS1)) - dot(vis, QM) --> ps_task = V, ps_state = TRANS0']  # noqa
            fi_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, SEV)) --> ps_task = F',  # noqa
                 '0.5 * (dot(ps_task, F-DEC) + dot(ps_state, TRANS0)) - dot(vis, QM) --> ps_task = F, ps_state = TRANS1',  # noqa
                 '0.5 * (dot(ps_task, F-DEC) + dot(ps_state, TRANS1)) - dot(vis, QM) --> ps_task = F, ps_state = TRANS2',  # noqa
                 '0.5 * (dot(ps_task, F-DEC) + dot(ps_state, TRANS2)) - dot(vis, QM) --> ps_task = F, ps_state = TRANS0']  # noqa
            decode_action = \
                ['dot(vis, QM) - 0.75 * dot(ps_task, W + C + V + F) --> ps_task = DEC, ps_state = ps_state, ps_dec = ps_dec',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, W)) --> ps_task = DEC, ps_state = ps_state, ps_dec = DECW',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, C)) --> ps_task = DEC, ps_state = ps_state, ps_dec = CNT',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, V + F)) --> ps_task = DEC, ps_state = ps_state, ps_dec = DECI',  # noqa
                 'dot(ps_task, DEC) - dot(ps_dec, CNT) --> ps_task = ps_task, ps_state = ps_state, ps_dec = ps_dec']  # noqa
                # ['dot(vis, QM) - 0.5 * dot(ps_task, C) --> ps_task = DEC, ps_state = ps_state, ps_dec = ps_dec',  # noqa
                #  '0.5 * (dot(vis, QM) + dot(ps_task, C)) --> ps_task = C, ps_state = ps_state, ps_dec = CNT',  # noqa
                #  'dot(ps_task, DEC) --> ps_task = ps_task, ps_state = ps_state, ps_dec = ps_dec']  # noqa
            default_action = \
                []
                # ['0.4 --> ps_task = ps_task, ps_state = ps_state, ps_dec = ps_dec']  # noqa
                # ['0.4 --> ps_task = ps_task, ps_state = ps_state, ps_dec = ps_dec']  # noqa

            all_actions = (copy_draw_action + recog_action +
                           mem_action + count_action + qa_action + rvc_action +
                           fi_action + decode_action + default_action)

            actions = spa.Actions(*all_actions)
            model.bg = spa.BasalGanglia(actions=actions, input_synapse=0.008)
            model.thal = spa.Thalamus(model.bg, mutual_inhibit=1)

        # ----- Set up connections (and save record of modules) -----
        # model.modules = []
        if hasattr(model, 'vis'):
            model.vis.setup_connections(model)
            # model.modules.append(model.vis)
        if hasattr(model, 'ps'):
            model.ps.setup_connections(model)
            # model.modules.append(model.ps)
        if hasattr(model, 'enc'):
            model.enc.setup_connections(model)
            # model.modules.append(model.enc)
        if hasattr(model, 'mem'):
            model.mem.setup_connections(model)
            # model.modules.append(model.mem)
        if hasattr(model, 'trfm'):
            model.trfm.setup_connections(model)
            # model.modules.append(model.trfm)
        if hasattr(model, 'dec'):
            model.dec.setup_connections(model)
            # model.modules.append(model.dec)
        if hasattr(model, 'mtr'):
            model.mtr.setup_connections(model)
            # model.modules.append(model.mtr)
        if hasattr(model, 'bg'):
            pass
            # model.modules.append(model.bg)
        if hasattr(model, 'thal'):
            pass
            # model.modules.append(model.bg)

    return model
