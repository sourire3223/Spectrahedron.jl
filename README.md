# Spectrahedron

Optimization on the set of density matrices (the spectrahedron) in Julia.

# Get Started
## Requirements
- Julia 1.11.7 or later
- The dependencies in `Project.toml` and `Manifest.toml` will be installed automatically when you run the code, or run
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
## Run
- For maximum likelihood quantum state tomography (MLQST) experiments, run
```
julia --project=. ./src/mlqst/main.jl
```