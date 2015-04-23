import nengo
from nengo import spa

from .config import cfg
from _spaun.modules import Stimulus, Vision, ProdSys, InfoEnc, InfoDec, Motor
from _spaun.modules import TrfmSys
# from _spaun.modules import Memory
from _spaun.modules.working_memory import WorkingMemoryDummy as Memory
from _spaun.modules.transform_system import TransformationSystemDummy as TrfmSys  # noqa


def Spaun():
    model = spa.SPA(label='Spaun', seed=cfg.seed)
    with model:
        model.config[nengo.Ensemble].max_rates = cfg.max_rates
        model.config[nengo.Ensemble].neuron_type = cfg.neuron_type
        model.config[nengo.Ensemble].n_neurons = cfg.n_neurons_ens
        model.config[nengo.Connection].synapse = cfg.pstc

        model.stim = Stimulus()
        model.vis = Vision()
        # model.ps = ProdSys()
        # model.enc = InfoEnc()
        # model.mem = Memory()
        # model.trfm = TrfmSys()
        # model.dec = InfoDec()
        # model.mtr = Motor()

        if hasattr(model, 'vis') and hasattr(model, 'ps'):
            copy_draw_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, ZER)) --> ps_task = W',  # noqa
                 'dot(ps_task, W) - dot(vis, QM) --> ps_task = W, ps_state = ps_state']  # noqa
            recog_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, ONE)) --> ps_task = R',  # noqa
                 'dot(ps_task, R) - dot(vis, QM) --> ps_task = R, ps_state = ps_state']  # noqa
            mem_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, THR)) --> ps_task = M',  # noqa
                 'dot(ps_task, M) - 0.5 * dot(vis, F + R + QM) --> ps_task = M, ps_state = ps_state',  # noqa
                 '0.5 * (dot(ps_task, M) + dot(vis, F)) --> ps_task = M, ps_dec = FWD',  # noqa
                 '0.5 * (dot(ps_task, M) + dot(vis, R)) --> ps_task = M, ps_dec = REV']  # noqa
            count_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, FOR)) --> ps_task = C',  # noqa
                 '0.5 * (dot(ps_task, C) + dot(ps_state, TRANS0)) - 0.8 * dot(vis, QM) --> ps_task = C, ps_state = TRANS1',  # noqa
                 '0.5 * (dot(ps_task, C) + dot(ps_state, TRANS1)) - 0.8 * dot(vis, QM) --> ps_task = C, ps_state = CNT']  # noqa
            # Count action is incomplete!
            qa_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, FIV)) --> ps_task = A',  # noqa
                 'dot(ps_task, A) - 0.5 * dot(vis, M + P + QM) --> ps_task = A, ps_state = ps_state',  # noqa
                 '0.5 * (dot(ps_task, A) + dot(vis, M)) --> ps_task = M, ps_state = QAN',  # noqa
                 '0.5 * (dot(ps_task, A) + dot(vis, P)) --> ps_task = M, ps_state = QAP']  # noqa
            rvc_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, SIX)) --> ps_task = V',  # noqa
                 '0.5 * (dot(ps_task, V) + dot(ps_state, TRANS0)) - 0.8 * dot(vis, QM) --> ps_task = V, ps_state = TRANS1',  # noqa
                 '0.5 * (dot(ps_task, V) + dot(ps_state, TRANS1)) - 0.8 * dot(vis, QM) --> ps_task = V, ps_state = TRANS0']  # noqa
            fi_action = \
                ['0.5 * (dot(ps_task, X) + dot(vis, SEV)) --> ps_task = F',  # noqa
                 '0.5 * (dot(ps_task, F) + dot(ps_state, TRANS0)) - 0.8 * dot(vis, QM) --> ps_task = F, ps_state = TRANS1',  # noqa
                 '0.5 * (dot(ps_task, F) + dot(ps_state, TRANS1)) - 0.8 * dot(vis, QM) --> ps_task = F, ps_state = TRANS2',  # noqa
                 '0.5 * (dot(ps_task, F) + dot(ps_state, TRANS2)) - 0.8 * dot(vis, QM) --> ps_task = F, ps_state = TRANS0']  # noqa
            decode_action = \
                ['dot(vis, QM) - 0.5 * dot(ps_task, W + V + F) --> ps_task = DEC, ps_dec = ps_dec, ps_state = ps_state',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, W)) --> ps_task = DECW, ps_dec = ps_dec, ps_state = ps_state',  # noqa
                 '0.5 * (dot(vis, QM) + dot(ps_task, V + F)) --> ps_task = DECI, ps_dec = ps_dec, ps_state = ps_state']  # noqa
            default_action = \
                ['0.5 --> ps_task = ps_task, ps_state = ps_state, ps_dec = ps_dec']  # noqa

            all_actions = (copy_draw_action + recog_action +
                           mem_action + count_action + qa_action + rvc_action +
                           fi_action + decode_action + default_action)

            actions = spa.Actions(*all_actions)
            model.bg = spa.BasalGanglia(actions=actions)
            model.thal = spa.Thalamus(model.bg, mutual_inhibit=2)

        # ----- Set up connections (and save record of modules) -----
        model.modules = []
        if hasattr(model, 'vis'):
            model.vis.setup_connections(model)
            model.modules.append(model.vis)
        if hasattr(model, 'ps'):
            model.ps.setup_connections(model)
            model.modules.append(model.ps)
        if hasattr(model, 'enc'):
            model.enc.setup_connections(model)
            model.modules.append(model.enc)
        if hasattr(model, 'mem'):
            model.mem.setup_connections(model)
            model.modules.append(model.mem)
        if hasattr(model, 'trfm'):
            model.trfm.setup_connections(model)
            model.modules.append(model.trfm)
        if hasattr(model, 'dec'):
            model.dec.setup_connections(model)
            model.modules.append(model.dec)
        if hasattr(model, 'mtr'):
            model.mtr.setup_connections(model)
            model.modules.append(model.mtr)
        if hasattr(model, 'bg'):
            model.modules.append(model.bg)
        if hasattr(model, 'thal'):
            model.modules.append(model.bg)

    return model
