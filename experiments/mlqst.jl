
using Random
include("../src/batch_algorithms.jl")
include("../src/mlqst.jl")

Random.seed!(407)

q::Int64 = 2
d::Int64 = 2 ^ q
M::Int64 = 4 ^ q
N::Int64 = 10 * 4 ^ q
# ρ_true = get_w_state(q) # true density matrix
ρ_true = get_random_state(q) # true density matrix
POVM = get_pauli_povms_positive(q)
idx_obs = rand(1:M, N)
outcomes = measure(ρ_true, POVM, idx_obs)
data = generate_data(POVM, idx_obs, outcomes)


loss_func_ = (ρ::Hermitian{<:Complex}) -> loss_func(ρ, data)
gradient_ = (ρ::Hermitian{<:Complex}) -> gradient(ρ, data)
loss_and_gradient_ = (ρ::Hermitian{<:Complex}) -> loss_and_gradient(ρ, data)


n_epoch = 30
ρ_init = Hermitian(Matrix{ComplexF64}(I, d, d) / d)
_, result_pgd = PGD(ρ_init, n_epoch, loss_func_, gradient_, loss_and_gradient_)
_, result_emd = EMD(ρ_init, n_epoch, loss_func_, gradient_, loss_and_gradient_)
_, result_fw = FW(ρ_init, n_epoch, loss_func_, gradient_, loss_and_gradient_)
_, result_bwrgd = BWRGD(ρ_init, n_epoch, loss_func_, gradient_, loss_and_gradient_)

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
savefig("experiments/mlqst.png")