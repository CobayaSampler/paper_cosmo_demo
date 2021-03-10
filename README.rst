Source code for the cosmological use case presented in `arXiv:2005.05290 <https://arxiv.org/abs/2005.05290>`_
=============================================================================================================

:Author: Jesus Torrado and Antony Lewis

To run:

.. code:: bash

   $ cobaya-install delensed.yaml  # installs CAMB
   $ python make_delensed_like.py
   $ cobaya-run delensed.yaml  # or lensed.yaml
