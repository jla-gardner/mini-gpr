.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/introduction
   tutorials/model-optimisation
   tutorials/sparse-approx
   tutorials/kernels

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/models
   api/kernels
   api/opt




############
``mini-gpr``
############

.. raw:: html

    <div align="center">
        <img src="_static/1d-gpr.gif" alt="1D GPR" width="400">
        <p style="font-size: 11pt"><em>hyperparameter-optimised GPR models trained </br> on a toy 1D dataset of increasing size</em></p>

    </div>

``mini-gpr`` is a minimal reference implementation of `Gaussian Process Regression <https://en.wikipedia.org/wiki/Gaussian_process>`__ in pure ``NumPy``, made
primarily for `my own learning <https://jla-gardner.github.io/>`_.

Features of ``mini-gpr`` include:

- implementations of a full (:class:`~mini_gpr.models.GPR`) and low rank (:class:`~mini_gpr.models.SoR`) GPR models
- implementations of several common :doc:`kernels <tutorials/kernels>`
- :class:`~mini_gpr.models.Model` and :class:`~mini_gpr.kernels.Kernel` base classes for easy extension
- automated hyperparameter optimisation against a range of objectives via :class:`mini_gpr.opt.optimise_model`
- strong typing using the `jaxtyping <https://docs.kidger.site/jaxtyping/>`__ library

This documentation includes:

- a `tutorial <tutorials>`_ in the form of a collection of Jupyter notebooks
- a `complete API reference <api>`_ for this small, stand-alone package

