.. _api.lightgbm_learning_api:

============
Learning API
============

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding lightgbm method.
The third column is a flag for whether or not there is an implementation in Xorbits for
the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` stands
for partial (meaning some parameters may not be supported yet), and ``D`` stands for
default to lightgbm.

+---------------------------+--------------+------------------------+----------------------------------+
| ``xorbits.lightgbm``      | ``lightgbm`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------------+--------------+------------------------+----------------------------------+
| ``predict``               |              | Y                      |                                  |
+---------------------------+--------------+------------------------+----------------------------------+
| ``predict_proba methods`` |              | Y                      |                                  |
+---------------------------+--------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
