*****
Spaun
*****

Spaun is currently the world's largest functional brain model.
It is the main focus of Chapter 7 of `How to Build a Brain
<http://www.amazon.com/How-Build-Brain-Architecture-Architectures/dp/0199794545/>`_
by Chris Eliasmith.
Spaun first appeared in Science;
these documents also serve as a home
to supporting material for the original paper.

The Spaun model consists of about 2.5 million spiking neurons.
It has a single eye and an arm.
All input is raw images shown to the eye,
and all output is arm movements,
controlled directly by the brain.

.. topic:: Spaun performing several tasks

   .. raw:: html

      <iframe width="100%" height="400" src="https://www.youtube.com/embed/RrxmlbZa7C4" frameborder="0" allowfullscreen></iframe>

This project contains the Spaun [1]_ model,
updated for Nengo 2.0.

.. [1] Chris Eliasmith, Terrence C. Stewart, Xuan Choo, Trevor Bekolay,
   Travis DeWolf, Yichuan Tang, and Daniel Rasmussen. A large-scale model
   of the functioning brain. Science, 338:1202-1205, 2012.
   doi:10.1126/science.1225266.

Running with Nengo OCL
======================

If you want to run with ``nengo_ocl``::

  python run_spaun.py -d 512 --ocl --ocl_platform=1 --ocl_device=3

where:

- the ``-d`` flag sets the dimensionality of Spaun,
- the ``--ocl`` flag tells the run script to use ``nengo_ocl``
- the --ocl_platform flag tells it what OCL platform to use
- the --ocl_device flag tells it what ocl device to use on said platform
  (this flag is optional, it's used in the context creation for pyopencl)

To determine the ``ocl_platform`` and ``ocl_device`` of the device you want to
use, see ``pyopencl.create_some_context()``.

To enable OCL profiling, find where the ``nengo_ocl.Simulator`` is created
in ``run_spaun.py``, and uncomment the version that has provifiling enabled.
Also uncomment the line to print profiling.

Resources
=========

.. toctree::
   :maxdepth: 2

   videos
   press
