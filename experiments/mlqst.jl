
using Dates
using Random
using Printf
using PyPlot

include("../src/batch-linesearch.jl")
include("../src/stochastic.jl")
include("../src/mlqst.jl")
include("../src/mlqst-bm.jl")
include("../src/utils.jl")


## Configurations
const seed = 407
const q::Int64 = 2
const d::Int64 = 2 ^ q
const M::Int64 = 4 ^ q
const N::Int64 = 100 * 4 ^ q
Random.seed!(seed)
const ρ_true = get_w_state(q) # true density matrix
# const ρ_true = get_random_state(q) # true density matrix

const folder =
	"./experiments/mlqst-$(q)w-" * Dates.format(now(), "yyyy-mm-dd-HH-MM-SS") * "/"
isdir(folder) || mkdir(folder)
# write setup to yaml
problem_params = Dict(
	"n_qubits" => q,
	"dim" => d,
	"n_observables" => M,
	"n_samples" => N,
	"true_state" => "W state",
	"seed" => seed,
)
using YAML
open(folder * "config.yaml", "w") do io
	YAML.write(io, problem_params)
end


## Problem setup
problem_setup_time = @elapsed begin
	# POVMs
	# const POVMs = get_pauli_povms(q) # Pauli POVMs
	const POVMs_positive = get_pauli_povms_positive(q)
	const idx_obs = rand(1:M, N)
	const outcomes = measure(ρ_true, POVMs_positive, idx_obs)
	const data = generate_data(POVMs_positive, idx_obs, outcomes)
	const frequency, POVMs = generate_data_freq(POVMs_positive, idx_obs, outcomes)


	# loss_func_ = (ρ::Hermitian{<:Complex}) -> @inline loss_func(ρ, data)
	# gradient_ = (ρ::Hermitian{<:Complex}) -> @inline gradient(ρ, data)
	# loss_and_gradient_ = (ρ::Hermitian{<:Complex}) -> @inline loss_and_gradient(ρ, data)
	# loss_func_bm_ = (halved::Matrix{<:Complex}) -> @inline loss_func_bm(halved, data)
	# gradient_bm_ = (halved::Matrix{<:Complex}) -> @inline gradient_bm(halved, data)
	# loss_and_gradient_bm_ =
	# 	(halved::Matrix{<:Complex}) -> @inline loss_and_gradient_bm(halved, data)

	loss_func_ = (ρ::Hermitian{<:Complex}) -> @inline loss_func(ρ, frequency, POVMs)
	gradient_ = (ρ::Hermitian{<:Complex}) -> @inline gradient(ρ, frequency, POVMs)
	loss_and_gradient_ =
		(ρ::Hermitian{<:Complex}) -> @inline loss_and_gradient(ρ, frequency, POVMs)
	loss_func_bm_ =
		(halved::Matrix{<:Complex}) -> @inline loss_func_bm(halved, frequency, POVMs)
	gradient_bm_ =
		(halved::Matrix{<:Complex}) -> @inline gradient_bm(halved, frequency, POVMs)
	loss_and_gradient_bm_ =
		(halved::Matrix{<:Complex}) ->
			@inline loss_and_gradient_bm(halved, frequency, POVMs)

end
@printf("Problem setup time: %.6f seconds\n", problem_setup_time)


n_epochs = 200
ρ_init = Hermitian(Matrix{ComplexF64}(I, d, d) / d)
halved_init1 = randn(ComplexF64, d, 1)
halved_init1 ./= norm(halved_init1)
halved_init4 = randn(ComplexF64, d, 4)
halved_init4 ./= norm(halved_init4)
halved_init16 = randn(ComplexF64, d, 16)
halved_init16 ./= norm(halved_init16)

# runtime dispatch
warmup_time = @elapsed begin
	projected_gradient_descent(
		ρ_init,
		1,
		loss_func_,
		gradient_,
		loss_and_gradient_;
		armijo_params = (α0 = 1e3, r = 0.5, τ = 0.8),
	)
	entropic_mirror_descent(
		ρ_init,
		1,
		loss_func_,
		gradient_,
		loss_and_gradient_;
		armijo_params = (α0 = 1e3, r = 0.5, τ = 0.8),
	)
	frank_wolfe(
		ρ_init,
		1,
		loss_func_,
		gradient_,
		loss_and_gradient_;
		armijo_params = (α0 = 0.99, r = 0.5, τ = 0.8),
	)
	bw_rgd(
		ρ_init,
		1,
		loss_func_,
		gradient_,
		loss_and_gradient_;
		armijo_params = (α0 = 1e3, r = 0.5, τ = 0.8),
	)
	sphere_rgd(
		halved_init1,
		1,
		loss_func_bm_,
		gradient_bm_,
		loss_and_gradient_bm_;
		armijo_params = (α0 = 1e3, r = 0.5, τ = 0.8),
	)
	sphere_rgd(
		halved_init4,
		1,
		loss_func_bm_,
		gradient_bm_,
		loss_and_gradient_bm_;
		armijo_params = (α0 = 1e3, r = 0.5, τ = 0.8),
	)
	sphere_rgd(
		halved_init16,
		1,
		loss_func_bm_,
		gradient_bm_,
		loss_and_gradient_bm_;
		armijo_params = (α0 = 1e3, r = 0.5, τ = 0.8),
	)
	sphere_srgd(halved_init16, 1, data, loss_func_bm, gradient_bm)
end
@printf("Warm-up time: %.6f seconds\n", warmup_time)




# Define experiments as (name, function, arguments..., kwargs)
experiments = [
	(
		name = "PGD",
		algo = projected_gradient_descent,
		args = (ρ_init, n_epochs, loss_func_, gradient_, loss_and_gradient_),
		kwargs = Dict(:armijo_params => (α0 = 120.0, r = 0.5, τ = 0.5)),
	),
	(
		name = "EMD",
		algo = entropic_mirror_descent,
		args = (ρ_init, n_epochs, loss_func_, gradient_, loss_and_gradient_),
		kwargs = Dict(:armijo_params => (α0 = 1000.0, r = 0.5, τ = 0.5)), # don't exceed 709 since exp(710) = Inf
	),
	(
		name = "FW",
		algo = frank_wolfe,
		args = (ρ_init, n_epochs, loss_func_, gradient_, loss_and_gradient_),
		kwargs = Dict(:armijo_params => (α0 = 1 - 1e-12, r = 0.8, τ = 0.5)),
	),
	(
		name = "BW-RGD",
		algo = bw_rgd,
		args = (ρ_init, n_epochs, loss_func_, gradient_, loss_and_gradient_),
		kwargs = Dict(:armijo_params => (α0 = 1e2, r = 0.5, τ = 0.2)),
	),
	(
		name = "Sphere-RGD-1",
		algo = sphere_rgd,
		args = (halved_init1, n_epochs, loss_func_bm_, gradient_bm_, loss_and_gradient_bm_),
		kwargs = Dict(:armijo_params => (α0 = 1e2, r = 0.5, τ = 0.2)),
	),
	(
		name = "Sphere-RGD-4",
		algo = sphere_rgd,
		args = (halved_init4, n_epochs, loss_func_bm_, gradient_bm_, loss_and_gradient_bm_),
		kwargs = Dict(:armijo_params => (α0 = 1e2, r = 0.5, τ = 0.2)),
	),
	(
		name = "Sphere-RGD-16",
		algo = sphere_rgd,
		args = (
			halved_init16,
			n_epochs,
			loss_func_bm_,
			gradient_bm_,
			loss_and_gradient_bm_,
		),
		kwargs = Dict(:armijo_params => (α0 = 1e2, r = 0.5, τ = 0.2)),
	),
	(
		name = "Sphere-SRGD-16",
		algo = sphere_srgd,
		args = (halved_init16, 5, data, loss_func_bm, gradient_bm),
		kwargs = Dict(:stepsize => 100.0),
	),
]


# Run algorithms

io = open(folder * "record.txt", "a")
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

# plot
figure(1)
for (name, result) in results
	semilogy(result["n_epoch"], result["fval"] .- min_loss, label = name)
end
legend()
xlabel("Number of Epochs")
ylabel("Error")
grid("on")
savefig(folder * "epoch-error.png")



figure(2)
for (name, result) in results
	semilogy(result["elapsed_time"], result["fval"] .- min_loss, label = name)
end
legend()
xlabel("Elapsed Time (s)")
ylabel("Error")
grid("on")
savefig(folder * "time-error.png")

close("all")