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