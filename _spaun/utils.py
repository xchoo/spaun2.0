import os
import time

from multiprocessing import Process, Array, Value
import numpy as np

from .configurator import cfg
from .experimenter import experiment
from .loggerator import logger
from .vocabulator import vocab


def get_total_n_neurons(model):
    return sum([e.n_neurons for e in model.all_ensembles])


def sum_vocab_vecs(vocab, vocab_strs):
    result = vocab[vocab_strs[0]].copy()

    for sp in vocab_strs[1:]:
        result += vocab[sp]

    return result.v


def conf_interval(data, num_samples=5000, confidence=0.95):
    mean_data = np.zeros(num_samples)

    for i in range(num_samples):
        mean_data[i] = np.mean(np.random.choice(data, len(data)))

    mean_data = np.sort(mean_data)

    low_ind = int(num_samples * (1 - confidence) * 0.5)
    high_ind = num_samples - low_ind - 1

    return (np.mean(data), mean_data[low_ind], mean_data[high_ind])


def strs_to_inds(str_list, ref_str_list):
    return [ref_str_list.index(s) for s in str_list]


def str_to_bool(string):
    return string.lower() in ["yes", "true", "t", "1"]


def invol_matrix(dim):
    result = np.eye(dim)
    return result[-np.arange(dim), :]


def get_probe_data_filename(label="probe_data", suffix="", ext="npz"):
    suffix = str(suffix).replace("?", "@")

    raw_seq = experiment.raw_seq_str.replace("?", "@").replace(":", ";")
    raw_seq = raw_seq.replace(">", ")").replace("<", "(")

    if experiment.present_blanks:
        raw_seq = "-".join(raw_seq)

    return "+".join([label,
                     "_".join([str(type(cfg.neuron_type).__name__),
                               str(vocab.sp_dim)]),
                     raw_seq[:150],
                     str(cfg.seed)]) + \
           ("" if suffix == "" else "(" + suffix + ")") + "." + ext


def validate_num_gpus(num_processes, cl_platform):
    def get_num_gpus(num_gpus, cl_platform):
        import pyopencl as cl
        num_gpus.value = len(cl.get_platforms()[cl_platform].get_devices())

    num_gpus = Value("i", 0)
    p = Process(target=get_num_gpus, args=(num_gpus, cl_platform))
    p.start()
    p.join()
    print(f"GPUs > REQUESTED: {num_processes} | AVAILABLE: {num_gpus.value}")

    if num_processes > num_gpus.value:
        raise RuntimeError("Error - Insufficient number of GPUs to run" +
                           " in multi-process mode." +
                           f"\nRequired number of GPUs: {num_processes}" +
                           f"\nAvailable number of GPUs: {num_gpus.value}")


def build_and_run_spaun_network(
        network, args, runtime, buildtimes, walltimes, proc_ind=-1
    ):
    # ----- Set up probes -----
    from _spaun import probes as probe_module

    make_probes = not args.no_probes
    if runtime > probe_module.max_probe_time and make_probes:
        print(">>> !!! WARNING !!! EST RUNTIME > %0.2fs - DISABLING PROBES" %
              max_probe_time)
        make_probes = False

    if proc_ind < 0:
        probe_data_filename = cfg.probe_data_filename
        print_prefix = ""
    else:
        probe_data_filename = f"{cfg.probe_data_filename}_{proc_ind}"
        print_prefix = f"[{proc_ind}] "

    if make_probes:
        print(print_prefix + "PROBE FILENAME: %s" % (probe_data_filename))
        probe_graph_config = getattr(probe_module, cfg.probe_graph_config)
        probe_cfg = probe_graph_config(network, vocab, cfg.sim_dt,
                                       cfg.data_dir, probe_data_filename)

    # ----- Set up animation probes -----
    if args.showanim or args.showiofig or args.probeio:
        anim_probe_data_filename = probe_data_filename[:-4] + "_anim.npz"
        probe_anim_config = getattr(probe_module, cfg.probe_anim_config)
        print(print_prefix + "ANIM PROBE FILENAME: %s" % anim_probe_data_filename)
        probe_anim_cfg = probe_anim_config(network, vocab, cfg.sim_dt,
                                           cfg.data_dir, anim_probe_data_filename)

    # ----- Spaun simulation build -----
    print(print_prefix + "START BUILD")
    t_build_start = time.time()

    if args.nengo_gui:
        # Set environment variables (for nengo_gui)
        if cfg.use_opencl:
            os.environ["PYOPENCL_CTX"] = "%s:%s" % (args.ocl_platform,
                                                    args.ocl_device)

        import sys
        import threading
        import webbrowser
        import nengo_gui

        host = "localhost"
        port = 8080 + max(proc_ind, 0)

        print(print_prefix + f"STARTING NENGO_GUI @ {host}:{port}")
        # gui = nengo_gui.GUI(__file__, model=network, locals=locals(), editor=False)

        gui = nengo_gui.InteractiveGUI(
            nengo_gui.guibackend.ModelContext(
                filename=sys.argv[0], model=network, locals=locals()
            ),
            page_settings=nengo_gui.page.PageSettings(
                editor_class=nengo_gui.components.editor.NoEditor
            ),
            server_settings=nengo_gui.guibackend.GuiServerSettings((host, port))
        )

        t = threading.Thread(
            target=webbrowser.open, args=(str(gui.server.get_url(token="one_time")),)
        )
        t.start()
        gui.start()

        print(print_prefix + "NENGO_GUI STOPPED")

        buildtimes[proc_ind] = -1
        return

    if cfg.use_opencl:
        import pyopencl as cl
        import nengo_ocl

        print(print_prefix + "------ OCL ------")

        print(print_prefix + "AVAILABLE (* USING) OCL PLATFORMS:")
        for i, pltf in enumerate(cl.get_platforms()):
            if i == args.ocl_platform:
                prefix = "   * "
            else:
                prefix = "   - "
            print(prefix + str(cl.get_platforms()[i]))
        pltf = cl.get_platforms()[args.ocl_platform]

        print(print_prefix + "AVAILABLE (* USING) OCL DEVICES:")
        for i, dev in enumerate(pltf.get_devices()):
            if i == args.ocl_device + max(proc_ind, 0):
                prefix = "   * "
            else:
                prefix = "   - "
            print(prefix + str(pltf.get_devices()[i]))

        ctx = cl.Context([pltf.get_devices()[args.ocl_device + max(proc_ind, 0)]])
        sim = nengo_ocl.Simulator(network, dt=cfg.sim_dt, context=ctx,
                                  profiling=args.ocl_profile)
    else:
        import nengo

        print(print_prefix + "------ REF ------")
        sim = nengo.Simulator(network, dt=cfg.sim_dt)

    t_build = time.time() - t_build_start
    buildtimes[proc_ind] = t_build
    print(print_prefix + f"BUILD FINISHED - build time: {t_build:.3f}s")

    # ----- Spaun simulation run -----
    experiment.reset()
    if cfg.use_opencl or cfg.use_ref:
        print(print_prefix + f"START SIM - est_runtime: {runtime:.3f}s")

        run_steps = int(runtime / cfg.sim_dt)

        if cfg.multi_process:
            # If using multiple processes, do one sim step first. This forces
            # process to wait until all other processes to start the simulation
            # before timing the run.
            sim.step()
            run_steps -= 1

        sim_start_time = time.time()
        sim.run_steps(run_steps)
        t_walltime = time.time() - sim_start_time
        walltimes[proc_ind] = t_walltime

        if args.ocl_profile:
            sim.print_plans()
            sim.print_profiling()

        print(print_prefix + "MODEL N_NEURONS: %i" % (get_total_n_neurons(network)))
        print(print_prefix + f"FINISHED! - Build time: {t_build:.3f}s, " +
              f"Sim runtime: {runtime:.3f}s, " +
              f"Wall time: {t_walltime:.3f}s")

    # ----- Close simulator -----
    if hasattr(sim, "close"):
        sim.close()

    # ----- Write probe data to file -----
    logger.write("\n\n# Command line options for displaying recorded probed " +
                 "data:")
    logger.write("\n# ------------------------------------------------------" +
                 "---")
    if make_probes:
        print("WRITING PROBE DATA TO FILE")
        probe_cfg.write_simdata_to_file(sim, experiment)

        # Assemble graphing subprocess call string
        subprocess_call_list = \
            ["python",
             os.path.join(cfg.cwd, "disp_probe_data.py"),
             f"\"{probe_data_filename}\"",
             "--data-dir",
             f"\"{cfg.data_dir}\""]

        # Log subprocess call
        logger.write("\n#\n# To display graphs of the recorded probe data:")
        logger.write("\n# > " + " ".join(subprocess_call_list + ["--showgrph"]))

        if args.showgrph:
            subprocess_call_list += ["--showgrph"]
        if args.savegrph:
            subprocess_call_list += ["--savegrph"]

        if args.showgrph or args.savegrph:
            # Open subprocess
            print(print_prefix + "CALLING: \n%s" % (" ".join(subprocess_call_list)))
            import subprocess
            subprocess.Popen(subprocess_call_list)

    if (args.showanim or args.showiofig or args.probeio):
        print(print_prefix + "WRITING ANIMATION PROBE DATA TO FILE")
        probe_anim_cfg.write_simdata_to_file(sim, experiment)

        # Assemble graphing subprocess call string
        subprocess_call_list = \
            ["python",
             os.path.join(cfg.cwd, "disp_probe_data.py"),
             f"\"{anim_probe_data_filename}\"",
             "--data-dir",
             f"\"{cfg.data_dir}\""]

        # Log subprocess call
        logger.write("\n#\n# To display Spaun's input/output plots:")
        logger.write("\n# > " + " ".join(subprocess_call_list +
                                         ["--showiofig"]))
        logger.write("\n#\n# To display Spaun's input/output animation:")
        logger.write("\n# > " + " ".join(subprocess_call_list +
                                         ["--showanim"]))
        logger.write("\n# (Flags can be combined to display both plots and" +
                     " animations)")

        if args.showanim:
            subprocess_call_list += ["--showanim"]
        if args.showiofig:
            subprocess_call_list += ["--showiofig"]

        if args.showanim or args.showiofig:
            # Open subprocess
            print(print_prefix + "CALLING: \n%s" % (" ".join(subprocess_call_list)))
            import subprocess
            subprocess.Popen(subprocess_call_list)
