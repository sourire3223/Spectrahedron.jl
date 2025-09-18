
using Dates
using Random
include("../src/batch_algorithms.jl")
include("../src/mlqst.jl")


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
p = Dict(
	"n_qubits" => q,
	"dim" => d,
	"n_observables" => M,
	"n_samples" => N,
	"true_state" => "W state",
	"seed" => seed,
)
using YAML
open(folder * "config.yaml", "w") do io
	YAML.write(io, p)
end


## Problem setup
const POVM = get_pauli_povms_positive(q)
const idx_obs = rand(1:M, N)
const outcomes = measure(ρ_true, POVM, idx_obs)
const data = generate_data(POVM, idx_obs, outcomes)


loss_func_ = (ρ::Hermitian{<:Complex}) -> loss_func(ρ, data)
gradient_ = (ρ::Hermitian{<:Complex}) -> gradient(ρ, data)
loss_and_gradient_ = (ρ::Hermitian{<:Complex}) -> loss_and_gradient(ρ, data)

io = open(folder * "record.txt", "a")
n_epoch = 200
ρ_init = Hermitian(Matrix{ComplexF64}(I, d, d) / d)
# runtime dispatch
armijo_params = ArmijoParams(0.99, 0.5, 0.8)
projected_gradient_descent(
	ρ_init,
	1,
	loss_func_,
	gradient_,
	loss_and_gradient_;
	armijo_params,
)
entropic_mirror_descent(ρ_init, 1, loss_func_, gradient_, loss_and_gradient_; armijo_params)
frank_wolfe(ρ_init, 1, loss_func_, gradient_, loss_and_gradient_; armijo_params)
bw_rgd(ρ_init, 1, loss_func_, gradient_, loss_and_gradient_; armijo_params)

# Run algorithms
armijo_params = ArmijoParams(10.0, 0.5, 0.5)
_, result_pgd = projected_gradient_descent(
	ρ_init,
	n_epoch,
	loss_func_,
	gradient_,
	loss_and_gradient_;
	armijo_params,
)
@printf(io, "PGD\n")
@printf(io, "%d\n", n_epoch)
write_output(io, result_pgd)

armijo_params = ArmijoParams(10.0, 0.5, 0.5)
_, result_emd = entropic_mirror_descent(
	ρ_init,
	n_epoch,
	loss_func_,
	gradient_,
	loss_and_gradient_;
	armijo_params,
)
@printf(io, "EMD\n")
@printf(io, "%d\n", n_epoch)
write_output(io, result_emd)

armijo_params = ArmijoParams(1 - 1e-12, 0.8, 0.5)
_, result_fw =
	frank_wolfe(ρ_init, n_epoch, loss_func_, gradient_, loss_and_gradient_; armijo_params)
@printf(io, "FW\n")
@printf(io, "%d\n", n_epoch)
write_output(io, result_fw)

armijo_params = ArmijoParams(1e2, 0.5, 0.1)
_, result_bwrgd =
	bw_rgd(ρ_init, n_epoch, loss_func_, gradient_, loss_and_gradient_; armijo_params)
@printf(io, "BW-RGD\n")
@printf(io, "%d\n", n_epoch)
write_output(io, result_bwrgd)
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
legend()
xlabel("Elapsed Time (s)")
ylabel("Error")
grid("on")
savefig(folder * "time-error.png")

close("all")