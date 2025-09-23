
using Dates
using Random
using Printf

include("../src/batch_algorithms.jl")
include("../src/mlqst.jl")
include("../src/mlqst_bm.jl")


## Configurations
const seed = 407
const q::Int64 = 6
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

	loss_func_ = (ρ::Hermitian{<:Complex}) -> loss_func(ρ, frequency, POVMs)
	gradient_ = (ρ::Hermitian{<:Complex}) -> gradient(ρ, frequency, POVMs)
	loss_and_gradient_ =
		(ρ::Hermitian{<:Complex}) -> loss_and_gradient(ρ, frequency, POVMs)
	loss_func_bm_ =
		(halved::Matrix{<:Complex}) -> loss_func_bm(halved, frequency, POVMs)
	gradient_bm_ = (halved::Matrix{<:Complex}) -> gradient_bm(halved, frequency, POVMs)
	loss_and_gradient_bm_ =
		(halved::Matrix{<:Complex}) -> loss_and_gradient_bm(halved, frequency, POVMs)
end
@printf("Problem setup time: %.6f seconds\n", problem_setup_time)


n_epoch = 200
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
end
@printf("Warm-up time: %.6f seconds\n", warmup_time)


# Run algorithms
io = open(folder * "record.txt", "a")

~, result_pgd = projected_gradient_descent(
	ρ_init,
	n_epoch,
	loss_func_,
	gradient_,
	loss_and_gradient_;
	armijo_params = (α0 = 10.0, r = 0.5, τ = 0.5),
)
@printf(io, "PGD\n")
@printf(io, "%d\n", n_epoch)
write_output(io, result_pgd)

~, result_emd = entropic_mirror_descent(
	ρ_init,
	n_epoch,
	loss_func_,
	gradient_,
	loss_and_gradient_;
	armijo_params = (α0 = 10.0, r = 0.5, τ = 0.5),
)
@printf(io, "EMD\n")
@printf(io, "%d\n", n_epoch)
write_output(io, result_emd)

~, result_fw = frank_wolfe(
	ρ_init,
	n_epoch,
	loss_func_,
	gradient_,
	loss_and_gradient_;
	armijo_params = (α0 = 1 - 1e-12, r = 0.8, τ = 0.5),
)
@printf(io, "FW\n")
@printf(io, "%d\n", n_epoch)
write_output(io, result_fw)

~, result_bwrgd = bw_rgd(
	ρ_init,
	n_epoch,
	loss_func_,
	gradient_,
	loss_and_gradient_;
	armijo_params = (α0 = 1e2, r = 0.5, τ = 0.2),
)
@printf(io, "BW-RGD\n")
@printf(io, "%d\n", n_epoch)
write_output(io, result_bwrgd)

_, result_spherergd1 = sphere_rgd(
	halved_init1,
	n_epoch,
	loss_func_bm_,
	gradient_bm_,
	loss_and_gradient_bm_;
	armijo_params = (α0 = 1e2, r = 0.5, τ = 0.2),
)
@printf(io, "Sphere-RGD-1\n")
@printf(io, "%d\n", n_epoch)
write_output(io, result_spherergd1)

~, result_spherergd4 = sphere_rgd(
	halved_init4,
	n_epoch,
	loss_func_bm_,
	gradient_bm_,
	loss_and_gradient_bm_;
	armijo_params = (α0 = 1e2, r = 0.5, τ = 0.2),
)
@printf(io, "Sphere-RGD-4\n")
@printf(io, "%d\n", n_epoch)
write_output(io, result_spherergd4)

~, result_spherergd16 = sphere_rgd(
	halved_init16,
	n_epoch,
	loss_func_bm_,
	gradient_bm_,
	loss_and_gradient_bm_;
	armijo_params = (α0 = 1e2, r = 0.5, τ = 0.2),
)
@printf(io, "Sphere-RGD-16\n")
@printf(io, "%d\n", n_epoch)
write_output(io, result_spherergd16)


close(io)


min_loss = minimum([
	minimum(result_pgd["fval"]),
	minimum(result_emd["fval"]),
	minimum(result_fw["fval"]),
	minimum(result_bwrgd["fval"]),
])
@printf("Minimum loss: %.4e\n", min_loss)

using PyPlot
figure(1)
semilogy(result_pgd["n_epoch"], result_pgd["fval"] .- min_loss, label = "PGD")
semilogy(result_emd["n_epoch"], result_emd["fval"] .- min_loss, label = "EMD")
semilogy(result_fw["n_epoch"], result_fw["fval"] .- min_loss, label = "FW")
semilogy(result_bwrgd["n_epoch"], result_bwrgd["fval"] .- min_loss, label = "BW-RGD")
semilogy(
	result_spherergd1["n_epoch"],
	result_spherergd1["fval"] .- min_loss,
	label = "Sphere-RGD",
)
semilogy(
	result_spherergd4["n_epoch"],
	result_spherergd4["fval"] .- min_loss,
	label = "Sphere-RGD-4",
)
semilogy(
	result_spherergd16["n_epoch"],
	result_spherergd16["fval"] .- min_loss,
	label = "Sphere-RGD-16",
)


legend()
xlabel("Number of Epochs")
ylabel("Error")
grid("on")
savefig(folder * "epoch-error.png")

figure(2)
semilogy(result_pgd["elapsed_time"], result_pgd["fval"] .- min_loss, label = "PGD")
semilogy(result_emd["elapsed_time"], result_emd["fval"] .- min_loss, label = "EMD")
semilogy(result_fw["elapsed_time"], result_fw["fval"] .- min_loss, label = "FW")
semilogy(result_bwrgd["elapsed_time"], result_bwrgd["fval"] .- min_loss, label = "BW-RGD")
semilogy(
	result_spherergd1["elapsed_time"],
	result_spherergd1["fval"] .- min_loss,
	label = "Sphere-RGD",
)
semilogy(
	result_spherergd4["elapsed_time"],
	result_spherergd4["fval"] .- min_loss,
	label = "Sphere-RGD-4",
)
semilogy(
	result_spherergd16["elapsed_time"],
	result_spherergd16["fval"] .- min_loss,
	label = "Sphere-RGD-16",
)

legend()
xlabel("Elapsed Time (s)")
ylabel("Error")
grid("on")
savefig(folder * "time-error.png")

close("all")