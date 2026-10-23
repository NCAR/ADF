# Conda environments

This directory holds conda environment files for running the ADF:

| File | Purpose |
|---|---|
| `conda_environment.yaml` | Standard ADF environment |
| `inform_environment.yml` | ADF plus the INFORM diagnostics (adds cookbook packages and uxarray) |

The steps below work for any of these files. Replace `<env_file>` with the
file you want and `<env_name>` with the name you want for the environment.

## Build the environment

Create it under your conda's default location, using the `name:` field from
the file:

```bash
conda env create -f env/<env_file>
```

Or give it your own name:

```bash
conda env create -n <env_name> -f env/<env_file>
```

Or install it at a specific path (useful on shared systems where your home
directory has a small quota):

```bash
conda env create -p /path/to/conda-envs/<env_name> -f env/<env_file>
```

`mamba env create ...` takes the same arguments and solves faster.

Solving and downloading can take a while (tens of minutes for the larger
environments). On HPC systems, run the build on a compute node or batch job
rather than a login node if your site asks for that.

## Activate it

```bash
conda activate <env_name>            # if created with -n or the default name
conda activate /path/to/conda-envs/<env_name>   # if created with -p
```

## Check it

```bash
python -c "import xarray, xesmf, uxarray; print('ok')"
```

(`uxarray` is only in environments that list it.)

## Update an existing environment after the file changes

```bash
conda env update -n <env_name> -f env/<env_file> --prune
```

`--prune` removes packages that are no longer listed. If the update can't be
solved, remove the environment and build it again:

```bash
conda env remove -n <env_name>
conda env create -n <env_name> -f env/<env_file>
```

## Record exactly what was installed

The environment files pin only the key packages, so a build months later can
pick up newer versions of everything else. To save the exact versions of a
working environment:

```bash
conda env export -n <env_name> > my_env_lock.yml
```

You can rebuild from that file the same way, but only on the same platform
(for example, linux-64).
