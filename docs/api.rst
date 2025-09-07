API Reference
#############

Models
======


.. autoclass:: mini_gpr.models.Model
    :members:

.. autoclass:: mini_gpr.models.GPR
    :show-inheritance:

.. autoclass:: mini_gpr.models.SoR
    :show-inheritance:

Kernels
=======


Base classes
------------

.. autoclass:: mini_gpr.kernels.Kernel
    :members:
    :special-members: __call__, __add__, __mul__, __pow__

.. autoclass:: mini_gpr.kernels.SumKernel
    :members:

.. autoclass:: mini_gpr.kernels.ProductKernel
    :members:

.. autoclass:: mini_gpr.kernels.PowerKernel
    :members:
    
Kernel implementations
----------------------

.. autoclass:: mini_gpr.kernels.RBF
    :members:

.. autoclass:: mini_gpr.kernels.DotProduct
    :members:

.. autoclass:: mini_gpr.kernels.Constant
    :members:

.. autoclass:: mini_gpr.kernels.Linear
    :members:

.. autoclass:: mini_gpr.kernels.Periodic
    :members:


Solvers
=======

.. autoclass:: mini_gpr.solvers.LinearSolver
    :members:

.. autoclass:: mini_gpr.solvers.vanilla
    :members:

.. autoclass:: mini_gpr.solvers.least_squares
    :members:


Model optimisation
==================

Objectives
----------

.. autoclass:: mini_gpr.opt.Objective
    :special-members: __call__

.. autoclass:: mini_gpr.opt.maximise_log_likelihood
    :members:

.. autoclass:: mini_gpr.opt.validation_set_mse
    :members:

.. autoclass:: mini_gpr.opt.validation_set_log_likelihood
    :members:

Optimising
----------

.. autoclass:: mini_gpr.opt.optimise_model
    :members:

