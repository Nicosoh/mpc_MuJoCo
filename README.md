### Notes
#### ACADOS Installation
For ACADOS installation, the commands listed at https://docs.acados.org/installation/index.html created the cmake files outside of the build directory while it should have been created inside. The commands below fixes that.
```
mkdir -p build
```
This command is required for the cmake files to go into the build directory. Edit flags as necessary.

```
cmake -B build -S . -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OSQP=ON -DACADOS_WITH_QPDUNES=ON
```

Then to run 
```
cd build
make install -j4
```

#### Adding ACADOS to UV .venv
Assuming that acados was installed in a sister folder to this directory, then add the python bindings to the .venv. In this case, to a uv project.

```
uv add ../acados/interfaces/acados_template
```

#### Allowing the script to find the ACADOS directory
```
export LD_LIBRARY_PATH=absolute_path_to/acados/lib
export ACADOS_SOURCE_DIR=absolute_path_to/acados
```