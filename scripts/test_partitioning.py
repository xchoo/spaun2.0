import nengo
from nengo import spa

nengo.log(debug=False)

# ----- Configurations -----
from _spaun.config import cfg
cfg.use_mpi = True
cfg.sp_dim = 512
cfg.neuron_type = nengo.LIFRate()

# ----- Seeeeeeeed -----
print "MODEL SEED: %i" % cfg.seed

# ----- Spaun imports -----
from _spaun.modules import Stimulus, Vision, ProdSys, InfoEnc, Memory, InfoDec
from _spaun.modules import Motor

# ----- Spaun proper -----
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
    model.dec = InfoDec()
    model.mtr = Motor()

    if hasattr(model, 'vis') and hasattr(model, 'ps'):
        actions = spa.Actions(
            'dot(vis, A) --> ps_task = X',
            '0.5 * (dot(ps_task, X) + dot(vis, ZER)) --> ps_task = W',
            'dot(ps_task, W) - dot(vis, QM) --> ps_task = W',
            '0.5 * (dot(ps_task, X) + dot(vis, ONE)) --> ps_task = R',
            'dot(ps_task, R) - dot(vis, QM) --> ps_task = R',
            'dot(vis, QM) - dot(ps_task, W) --> ps_task = DEC',
            '0.5 * (dot(vis, QM) + dot(ps_task, W)) --> ps_task = DECW')
        model.bg = spa.BasalGanglia(actions=actions)
        model.thal = spa.Thalamus(model.bg, mutual_inhibit=2)

# ----- Set up connections -----
with model:
    if hasattr(model, 'vis'):
        model.vis.setup_connections(model)
    if hasattr(model, 'ps'):
        model.ps.setup_connections(model)
    if hasattr(model, 'enc'):
        model.enc.setup_connections(model)
    if hasattr(model, 'mem'):
        model.mem.setup_connections(model)
    if hasattr(model, 'dec'):
        model.dec.setup_connections(model)
    if hasattr(model, 'mtr'):
        model.mtr.setup_connections(model)

num_components = 1024

from nengo_mpi.partition import work_balanced_partitioner
from nengo_mpi.partition import spectral_partitioner
from nengo_mpi.partition import top_level_partitioner
from nengo_mpi.partition import metis_partitioner
from nengo_mpi.partition import evaluate_partition, propogate_assignments


def test_work_balanced(model):
    print "\nwork_balanced_partitioner"
    assignments = work_balanced_partitioner(model, num_components)
    propogate_assignments(model, assignments)
    evaluate_partition(model, num_components, assignments)

test_work_balanced(model)


def test_spectral(model):
    print "\nspectral_partitioner"
    assignments = spectral_partitioner(model, num_components)
    propogate_assignments(model, assignments)
    evaluate_partition(model, num_components, assignments)

test_spectral(model)


# def test_top_level(model):
#     print "\ntop_level_partitioner"
#     assignments = top_level_partitioner(model, num_components)
#     propogate_assignments(model, assignments)
#     evaluate_partition(model, num_components, assignments)
#
# test_top_level(model)


def test_metis(model):
    print "\nmetis_partitioner"
    assignments = metis_partitioner(model, num_components)
    propogate_assignments(model, assignments)
    evaluate_partition(model, num_components, assignments)

test_metis(model)
