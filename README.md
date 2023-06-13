D+ - XRay Solution Scattering Analysis Tool
========================

D+ is an innovative program that computes the X-ray scattering from large supramolecular structures in solutions at any resolution down to (near) atomic resolution.
Using its different algorithms, we can dock atomic and/or geometric models into their assembly symmetry and do it in a hierarchical manner, in a bottom up approach, adding as many subunits as needed.
The assembly symmetry defines the rotations and translations of repeating subunits in a large assembly.
In this way, the solution scattering curve from any supramolecular structure can be modeled at any spatial resolution (including atomic). The solvation layer of the structures can also be computed in a scalable manner for large complexes.
Depending on the users' computer, D+ can be run on both CPU and GPU in a parallelized manner.

On top of the UI, a python-API is also available and can be used independently of the UI. It can be used together with other modules such as numpy and scipy, so as to build models bespoke to your needs.

Papers about D+ can be found [here](https://scholars.huji.ac.il/uriraviv/book/papers-about-d). Please cite those papers if you are using D+.

The source code of D+ is available for academic users and developers. 

D+ UI
-----
The latest version of D+ can be found [here](https://github.com/uri-raviv-lab/dplus-dev/releases).
Its manual can be found inside the desktop folder after installation or can be [downloaded](https://scholars.huji.ac.il/sites/default/files/uriraviv/files/dmanual.pdf).
On top of that, we also have [a few tutorials](https://scholars.huji.ac.il/uriraviv/book/tutorials-d) to get you started.

D+ API
------
The API can be downloaded using PIP by running the following inside the command prompt:

`pip install --extra-index-url https://pypi.fury.io/theresearchsoftwarecompany/ dplus-api`

A short tutorial with examples can be found [here](https://github.com/uri-raviv-lab/dplus-dev/tree/development/PythonInterface).

Hardware architecture.
----------------------
Computations and tests were performed on Windows computers with an Intel Core i5 3470 3.2 GHz CPU and an NVIDIA GeForce GTX Titan GPU card, or Intel Core i5 4590 3.3 GHz CPU and an NVIDIA GeForce GTX Titan Black GPU card. GeForce RTX 3080, GTX 1070, GTX 970, NVIDIA TESLA K80 and GTX 960 cards have also passed our tests.
In general, we believe most architectures built in the past years should work.

Troubleshooting
---------------
Questions, bugs, and problems can all be asked in our [issues section](https://github.com/uri-raviv-lab/dplus-dev/issues).
