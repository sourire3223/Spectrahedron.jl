
using Dates
using Random
using Printf
using PyPlot

include("../algorithms/batch-linesearch.jl")
include("../algorithms/batch.jl")
include("../algorithms/stochastic.jl")
include("../algorithms/second-order.jl")
include("../utils.jl")

include("./functions.jl")
include("./functions-bm.jl")
include("./alg-batch.jl")
include("./alg-stochastic.jl")

const PROJECT_ROOT = pwd()
const EXPERIMENTS = joinpath(PROJECT_ROOT, "experiments/mlqst/")


## ======== Problem Setup ========
const seed = 407
const q::Int64 = 2
const true_state = :w_state # :w_state, :ghz_state, :random_state
include("./setup.jl")
include("./output.jl")

prefix = "$q" * (true_state == :w_state ? "w" : true_state == :ghz_state ? "ghz" : "rand")

const FOLDER = EXPERIMENTS * prefix * "-" * Dates.format(now(), "yyyy-mm-dd-HH-MM-SS") * "/"
isdir(FOLDER) || mkdir(FOLDER)

# write setup to yaml
problem_params = Dict(
	"n_qubits" => q,
	"dim" => d,
	"n_observables" => M,
	"n_samples" => N,
	"true_state" => string(true_state),
	"seed" => seed,
)
using YAML
open(FOLDER * "config.yaml", "w") do io
	YAML.write(io, problem_params)
end

# ======== Experiments ========
n_epochs = 300
ρ_init = Hermitian(Matrix{ComplexF64}(I, d, d) / d)
halved_init1 = get_random_state_halved(q, 1)
halved_init4 = get_random_state_halved(q, 4)
halved_init16 = get_random_state_halved(q, 16)
halved_init_full = get_random_state_halved(q, d)



experiments = [
	# (
	# 	name = "RρR",
	# 	algo = RρR,
	# 	args = (ρ_init, n_epochs),
	# 	kwargs = Dict(
	# 		:loss_func => loss_func_,
	# 		:gradient => gradient_,
	# 		:loss_and_gradient => loss_and_gradient_,
	# 		:output_functions => output_functions_mlqst,
	# 	),
	# ),
	# (
	# 	name = "QEM",
	# 	algo = QEM_last,
	# 	args = (ρ_init, n_epochs),
	# 	kwargs = Dict(
	# 		:loss_func => loss_func_,
	# 		:gradient => gradient_,
	# 		:loss_and_gradient => loss_and_gradient_,
	# 		:output_functions => output_functions_mlqst,
	# 	),
	# ),
	(
		name = "PGD",
		algo = projected_gradient_descent,
		args = (ρ_init, n_epochs),
		kwargs = Dict(
			:loss_func => loss_func_,
			:gradient => gradient_,
			:loss_and_gradient => loss_and_gradient_,
			:armijo_params => (α0 = 120.0, r = 0.5, τ = 0.5),
			:output_functions => output_functions_mlqst,
		),
	),
	# (
	# 	name = "EMD",
	# 	algo = entropic_mirror_descent,
	# 	args = (ρ_init, n_epochs),
	# 	kwargs = Dict(
	# 		:loss_func => loss_func_,
	# 		:gradient => gradient_,
	# 		:loss_and_gradient => loss_and_gradient_,
	# 		:armijo_params => (α0 = 1000.0, r = 0.5, τ = 0.5),
	# 		:output_functions => output_functions_mlqst,
	# 	),
	# ),
	# (
	# 	name = "FW",
	# 	algo = frank_wolfe,
	# 	args = (ρ_init, n_epochs),
	# 	kwargs = Dict(
	# 		:loss_func => loss_func_,
	# 		:gradient => gradient_,
	# 		:loss_and_gradient => loss_and_gradient_,
	# 		:armijo_params => (α0 = 1 - 1e-8, r = 0.8, τ = 0.5),
	# 		:output_functions => output_functions_mlqst,
	# 	),
	# ),
	# (
	# 	name = "BW-RGD",
	# 	algo = bw_rgd,
	# 	args = (ρ_init, n_epochs),
	# 	kwargs = Dict(
	# 		:loss_func => loss_func_,
	# 		:gradient => gradient_,
	# 		:loss_and_gradient => loss_and_gradient_,
	# 		:armijo_params => (α0 = 1e2, r = 0.5, τ = 0.2),
	# 		:output_functions => output_functions_mlqst,
	# 	),
	# ),
	# (
	# 	name = "Sphere-RGD (r=4)",
	# 	algo = sphere_rgd,
	# 	args = (halved_init4, n_epochs),
	# 	kwargs = Dict(
	# 		:loss_func => loss_func_bm_,
	# 		:gradient => gradient_bm_,
	# 		:loss_and_gradient => loss_and_gradient_bm_,
	# 		:armijo_params => (α0 = 1e2, r = 0.5, τ = 0.2),
	# 		:output_functions => output_functions_mlqst,
	# 	),
	# ),
	(
		name = "Q-Soft-Bayes",
		algo = q_soft_bayes,
		args = (ρ_init, 50),
		kwargs = (
			:data => data,
			:loss_func => loss_func,
			:gradient => gradient,
			:output_functions => output_functions_mlqst,
		),
	),
	(
		name = "Sphere-SRGD (a=1)",
		algo = sphere_srgd,
		args = (halved_init4, 50),
		kwargs = (
			:data => data,
			:loss_func => loss_func_bm,
			:gradient => gradient_bm,
			:stepsize => t -> 1.0 / (t + 1.0),
			:output_functions => output_functions_mlqst,
		),
	),
	(
		name = "Sphere-SRGD (a=100)",
		algo = sphere_srgd,
		args = (halved_init4, 50),
		kwargs = (
			:data => data,
			:loss_func => loss_func_bm,
			:gradient => gradient_bm,
			:stepsize => t -> 100.0 / (t + 100.0),
			:output_functions => output_functions_mlqst,
		),
	),
	(
		name = "Sphere-SRGD (a=1, γ=0.6)",
		algo = sphere_srgd,
		args = (halved_init4, 50),
		kwargs = (
			:data => data,
			:loss_func => loss_func_bm,
			:gradient => gradient_bm,
			:stepsize => t -> 1.0 / (t + 1.0) ^ 0.6,
			:output_functions => output_functions_mlqst,
		),
	),
	(
		name = "Sphere-SRGD (a=100, γ=0.6)",
		algo = sphere_srgd,
		args = (halved_init4, 50),
		kwargs = (
			:data => data,
			:loss_func => loss_func_bm,
			:gradient => gradient_bm,
			:stepsize => t -> 100.0 / (t + 100.0) ^ 0.6,
			:output_functions => output_functions_mlqst,
		),
	),
	# (
	# 	name = "trust-region-tCG (r=4)",
	# 	algo = trust_region_tCG,
	# 	args = (halved_init4, n_epochs),
	# 	kwargs = Dict(
	# 		:loss_func => loss_func_bm_,
	# 		:gradient => gradient_bm_,
	# 		:hessian => hessian_bm_,
	# 		:loss_and_gradient => loss_and_gradient_bm_,
	# 		:trust_region_params => (Δ̄ = 200.0, Δ₀ = 100.0, ρ′ = 0.0),
	# 		:output_functions => output_functions_mlqst,
	# 	),
	# ),
]
# Warm-up (runtime dispatch, JIT)
warmup_time = @elapsed begin
	for (name, algo, args, kwargs) in experiments
		_, result = algo(args[1], 1; kwargs...)
	end
end
@printf("Warm-up time: %.6f seconds\n", warmup_time)

# Run algorithms
io = open(FOLDER * "record.txt", "a")
results = []
for (name, algo, args, kwargs) in experiments
	_, result = algo(args...; kwargs...)
	@printf(io, "%s\n", name)
	@printf(io, "%d\n", args[2])  # n_epochs assumed to be 2nd argument
	write_output!(io, result)
	push!(results, name => result)
end

close(io)

# compute minimum
min_loss = minimum([minimum(result["fval"]; init = Inf) for (name, result) in results])
@printf("Minimum loss: %.4e\n", min_loss)

# ======== Plots ========
figure(1)
for (name, result) in results
	semilogy(result["n_epoch"], result["fval"] .- min_loss, label = name)
end
legend()
xlabel("Number of Epochs")
ylabel("Error")
grid("on")
savefig(FOLDER * "epoch-error.png")



figure(2)
for (name, result) in results
	semilogy(result["elapsed_time"], result["fval"] .- min_loss, label = name)
end
legend()
xlabel("Elapsed Time (s)")
ylabel("Error")
grid("on")
savefig(FOLDER * "time-error.png")



figure(3)
for (name, result) in results
	plot(result["n_epoch"], result["fidelity"], label = name)
end
legend()
xlabel("Number of Epochs")
ylabel("Fidelity")

grid("on")
savefig(FOLDER * "epoch-fidelity.png")



figure(4)
for (name, result) in results
	plot(result["elapsed_time"], result["fidelity"], label = name)
end
legend()
xlabel("Elapsed Time (s)")
ylabel("Fidelity")
grid("on")
savefig(FOLDER * "time-fidelity.png")



close("all")