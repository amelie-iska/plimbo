.. # ------------------( DIRECTIVES                         )------------------
.. # Fallback language applied to all code blocks failing to specify an
.. # explicit language. Since the majority of all code blocks in this document
.. # are Bash one-liners intended to be run interactively, this is "console".
.. # For a list of all supported languages, see also:
.. #     http://build-me-the-docs-please.readthedocs.org/en/latest/Using_Sphinx/ShowingCodeExamplesInSphinx.html#pygments-lexers

.. # FIXME: Sadly, this appears to be unsupported by some ReST parsers and hence
.. # is disabled until more widely supported. *collective shrug*
.. # highlight:: console

.. # ------------------( SYNOPSIS                           )------------------

======
PLIMBO
======

**PLIMBO** (**P**\ lanarian **I**\ nterface for **M**\ odelling **B**\ ody
**O**\ rganization) is an open-source cross-platform simulator of morphogen-
directed regeneration of planarian head and tail, using a model reported on
in the manuscript *Neural Control of Body-plan Axis in Regenerating Planaria.*
PLIMBO represents a 1D and 2D simulator of the reaction-diffusion-convection
model for planarian regeneration, with tools for facilitating model discovery
and sensitivity analysis.

PLIMBO is associated with the `Paul Allen Discovery Center`_ at `Tufts
University`_ and supported by a `Paul Allen Discovery Center award`_ from the
`Paul G. Allen Frontiers Group`_.

PLIMBO is `portably implemented <codebase_>`__ in pure `Python 3`_, will be
`continuously stress-tested <testing_>`__ with GitLab-CI_ **×** Appveyor_ **+**
py.test_, and `permissively distributed <License_>`__ under the `BSD 2-clause
license`_.

Installation
============

PLIMBO currently supports **Linux**, **macOS**, and **Windows** out-of-the-box:

- [\ *Windows*\ ] Emulate **Ubuntu Linux** via the `Windows Subsystem for Linux
  (WSL) <WSL_>`__. [#windows_not]_
- Install the **Python 3.x** [#python2_not]_ (e.g., 3.6) variant of Anaconda_.
- Open a **Bash terminal.** [#terminal]_
- Run the following commands.

  - Enable conda-forge_.

    .. code-block:: console

       conda config --add channels conda-forge

  - Install all mandatory dependencies of **PLIMBO.**

    .. code-block:: console

       conda install betse=0.9.2 scikit-learn=0.20.2

  - Download the `most recent stable release <tarballs_>`__ of **PLIMBO.**

    .. code-block:: console

       curl https://gitlab.com/betse/plimbo/-/archive/v0.0.1/plimbo-v0.0.1.tar.gz -o plimbo.tar.gz && tar -xvzf plimbo.tar.gz

  - Install **PLIMBO.**

    .. code-block:: console

       sudo ln -s plimbo-v0.0.1/plimbo "$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/"

.. [#windows_not]
   The `Windows Subsystem for Linux (WSL) <WSL_>`__ and – hence PLIMBO itself –
   is *only* installable under **Windows 10.** Under older Windows versions,
   PLIMBO may be installed from a `virtual Linux guest <VirtualBox_>`__.

.. [#python2_not]
   Do *not* install the **Python 2.7** variant of Anaconda_. PLIMBO requires
   **Python 3.x.**

.. [#terminal]
   To open a `POSIX`_\ -compatible terminal under:

   - **Windows:**

     #. Install **Ubuntu Linux** via the `Windows Subsystem for Linux (WSL) <WSL_>`__.
     #. Open an *Ubuntu Linux terminal.*

   - **macOS:**

     #. Open the *Finder*.
     #. Open the *Applications* folder.
     #. Open the *Utilities* folder.
     #. Open *Terminal.app*.

   - **Ubuntu Linux:**

     #. Type ``Ctrl``\ +\ ``Alt``\ +\ ``t``.

Usage
=====

To use PLIMBO, open the
`"plimbo-0.0.1/plimbo/Plimbo_Runner.ipynb" file extracted above <notebook_>`__
with `Jupyter Notebook`_, which comes preinstalled with Anaconda_. This notebook
serves as a high-level interface to the PLIMBO ecosystem, complete with detailed
documentation.

Contact
=======

To contact `Dr. Pietak`_, the principal developer of the PLIMBO codebase and
the PLIMBO planaria model, please direct correspondence to:

* The personal e-mail account of `Dr. Pietak`_: [#e-mail]_

  * *Username:* **alexis** ``{dot}`` **pietak**
  * *Hostname:* **gmail** ``{dot}`` **com**

To report a software issue (e.g., bug, crash, or other unexpected behaviour)
*or* request a new feature in PLIMBO, consider `submitting a new issue <issue
submission_>`__ to our `issue tracker`_. Thanks in advance.

.. [#e-mail]
   To protect Dr. Pietak's e-mail address against `automated harvesting <e-mail
   harvesting_>`__, this address has been intentionally obfuscated. Reconstruct
   the original address by:

   * Replacing the ``{dot}`` substring with the ``.`` charecter.
   * Concatenating the username and hostname strings with the ``@`` character.


License
=======

PLIMBO is open-source software `released <license_>`__ under the permissive
`BSD 2-clause license`_.

.. # ------------------( LINKS ~ plimbo                     )------------------
.. _author list:
   doc/md/AUTHORS.md
.. _codebase:
   https://gitlab.com/betse/plimbo/tree/master
.. _conda package:
   https://anaconda.org/conda-forge/plimbo
.. _contributors:
   https://gitlab.com/betse/plimbo/graphs/master
.. _dependencies:
   doc/md/INSTALL.md
.. _issue submission:
   https://gitlab.com/betse/plimbo/issues/new?issue%5Bassignee_id%5D=&issue%5Bmilestone_id%5D=
.. _issue tracker:
   https://gitlab.com/betse/plimbo/issues
.. _license:
   LICENSE
.. _notebook:
   Plimbo_Runner.ipynb
.. _project:
   https://gitlab.com/betse/plimbo
.. _testing:
   https://gitlab.com/betse/plimbo/pipelines
.. _tarballs:
   https://gitlab.com/betse/plimbo/tags

.. # ------------------( LINKS ~ academia                   )------------------
.. _Michael Levin:
.. _Levin, Michael:
   https://ase.tufts.edu/biology/labs/levin
.. _Channelpedia:
   http://channelpedia.epfl.ch
.. _Paul Allen Discovery Center:
   http://www.alleninstitute.org/what-we-do/frontiers-group/discovery-centers/allen-discovery-center-tufts-university
.. _Paul Allen Discovery Center award:
   https://www.alleninstitute.org/what-we-do/frontiers-group/news-press/press-resources/press-releases/paul-g-allen-frontiers-group-announces-allen-discovery-center-tufts-university
.. _Paul G. Allen Frontiers Group:
   https://www.alleninstitute.org/what-we-do/frontiers-group
.. _Tufts University:
   https://www.tufts.edu

.. # ------------------( LINKS ~ academia : ally            )------------------
.. _Alexis Pietak:
.. _Pietak, Alexis:
.. _Dr. Pietak:
   https://www.researchgate.net/profile/Alexis_Pietak
.. _Organic Mechanics:
   https://www.omecha.org
.. _Organic Mechanics Contact:
   https://www.omecha.org/contact

.. # ------------------( LINKS ~ science                    )------------------
.. _bioelectricity:
   https://en.wikipedia.org/wiki/Bioelectromagnetics
.. _biochemical reaction networks:
   http://www.nature.com/subjects/biochemical-reaction-networks
.. _electrodiffusion:
   https://en.wikipedia.org/wiki/Nernst%E2%80%93Planck_equation
.. _electro-osmosis:
   https://en.wikipedia.org/wiki/Electro-osmosis
.. _enzyme activity:
   https://en.wikipedia.org/wiki/Enzyme_assay
.. _ephaptic coupling:
   https://en.wikipedia.org/wiki/Ephaptic_coupling
.. _epigenetics:
   https://en.wikipedia.org/wiki/Epigenetics
.. _extracellular environment:
   https://en.wikipedia.org/wiki/Extracellular
.. _finite volume:
   https://en.wikipedia.org/wiki/Finite_volume_method
.. _galvanotaxis:
   https://en.wiktionary.org/wiki/galvanotaxis
.. _gap junction:
.. _gap junctions:
   https://en.wikipedia.org/wiki/Gap_junction
.. _gene products:
   https://en.wikipedia.org/wiki/Gene_product
.. _gene regulatory networks:
   https://en.wikipedia.org/wiki/Gene_regulatory_network
.. _genetics:
   https://en.wikipedia.org/wiki/Genetics
.. _genetic algorithms:
   https://en.wikipedia.org/wiki/Genetic_algorithm
.. _Hodgkin-Huxley (HH) formalism:
   https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model
.. _local field potentials:
   https://en.wikipedia.org/wiki/Local_field_potential
.. _membrane permeability:
   https://en.wikipedia.org/wiki/Cell_membrane
.. _resting potential:
   https://en.wikipedia.org/wiki/Resting_potential
.. _tight junctions:
   https://en.wikipedia.org/wiki/Tight_junction
.. _transmembrane voltage:
   https://en.wikipedia.org/wiki/Membrane_potential
.. _transepithelial potential:
   https://en.wikipedia.org/wiki/Transepithelial_potential_difference

.. # ------------------( LINKS ~ science : computer         )------------------
.. _Big Data:
   https://en.wikipedia.org/wiki/Big_data
.. _comma-separated values:
   https://en.wikipedia.org/wiki/Comma-separated_values
.. _continuous integration:
   https://en.wikipedia.org/wiki/Continuous_integration
.. _directed graphs:
   https://en.wikipedia.org/wiki/Directed_graph
.. _e-mail harvesting:
   https://en.wikipedia.org/wiki/Email_address_harvesting
.. _genenic algorithms:
   https://en.wikipedia.org/wiki/Genetic_algorithm
.. _knowledge-based systems:
   https://en.wikipedia.org/wiki/Knowledge-based_systems

.. # ------------------( LINKS ~ os : linux                 )------------------
.. _APT:
   https://en.wikipedia.org/wiki/Advanced_Packaging_Tool
.. _POSIX:
   https://en.wikipedia.org/wiki/POSIX
.. _Ubuntu:
.. _Ubuntu Linux:
   https://www.ubuntu.com
.. _Ubuntu Linux 16.04 (Xenial Xerus):
   http://releases.ubuntu.com/16.04

.. # ------------------( LINKS ~ os : macos                 )------------------
.. _Homebrew:
   http://brew.sh
.. _MacPorts:
   https://www.macports.org

.. # ------------------( LINKS ~ os : windows               )------------------
.. _WSL:
   https://msdn.microsoft.com/en-us/commandline/wsl/install-win10

.. # ------------------( LINKS ~ software                   )------------------
.. _Appveyor:
   https://ci.appveyor.com/project/betse/plimbo/branch/master
.. _Atom:
   https://atom.io
.. _dill:
   https://pypi.python.org/pypi/dill
.. _FFmpeg:
   https://ffmpeg.org
.. _Git:
   https://git-scm.com/downloads
.. _GitLab-CI:
   https://about.gitlab.com/gitlab-ci
.. _Graphviz:
   http://www.graphviz.org
.. _imageio:
   https://imageio.github.io
.. _Jupyter Notebook:
   https://jupyter.org
.. _Libav:
   https://libav.org
.. _Matplotlib:
   http://matplotlib.org
.. _NumPy:
   http://www.numpy.org
.. _MEncoder:
   https://en.wikipedia.org/wiki/MEncoder
.. _Python 3:
   https://www.python.org
.. _py.test:
   http://pytest.org
.. _SciPy:
   http://www.scipy.org
.. _VirtualBox:
   https://www.virtualbox.org
.. _YAML:
   http://yaml.org

.. # ------------------( LINKS ~ software : conda           )------------------
.. _Anaconda:
   https://www.anaconda.com/download
.. _Anaconda packages:
   https://anaconda.org
.. _conda-forge:
   https://conda-forge.org

.. # ------------------( LINKS ~ software : licenses        )------------------
.. _license compatibility:
   https://en.wikipedia.org/wiki/License_compatibility#Compatibility_of_FOSS_licenses
.. _BSD 2-clause license:
   https://opensource.org/licenses/BSD-2-Clause
.. _CC BY 3.0 license:
   https://creativecommons.org/licenses/by/3.0
