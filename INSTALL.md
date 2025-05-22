# Installing geoPFA

There are several ways to install a Python package.
We strongly recommend using a virtual environment to avoid conflicts with other packages and projects in your system.
For development we use `pixi` and is our recommended way of installing it.

Some options are:

1. [Installing with PIP](#installing-with-pip)
2. [Installing with wheels](#installing-with-wheels)
3. [Installing with Pixi (recommended)](#installing-with-pixi-recommended)
4. [Installing with Conda](#installing-with-conda)

### Installing with PIP

### Installing from released Wheels

Download from https://github.com/GeothermalExplorationTools/geopfa/releases

### Installing with Pixi (recommended)

We recommend using `pixi` to manage a Python environment with `geoPFA` and all the required dependencies since it can reproduce exactly what we use for development.

1. Install `pixi` itself on your system. You only need to do this once, so you can skip this step if you already have `pixi` installed.

Please follow the instructions at [pixi.sh](https://pixi.sh) to install `pixi` on your system.

For Linux and MacOS:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh
   ```

For Windows:
   ```cmd
   powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
   ```

2. Clone geoPFA repository if you don't already have it.

3. Pixi provides support to use virtual environment with all the dependencies
   required. To do so, pixi uses the information from `pyproject.toml` and
   `pixi.lock` files, therefore, you need to move to where you cloned the
   repository. Once there, you can activate your environment by running:
   ```bash
   pixi shell
   ```

   You might notice a `(geoPFA)` prefix in your terminal, which
   indicates that you are now in the `geoPFA` environment.

   ```python
   import geopfa
   ```

### Installing with Conda
