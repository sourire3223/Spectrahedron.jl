using Random
using Printf

include("./functions.jl")
include("./functions-bm.jl")


## Configurations
# const seed = 407
# const q::Int64 = 4
const d::Int64 = 2 ^ q
const M::Int64 = 4 ^ q
const N::Int64 = 100 * 4 ^ q
Random.seed!(seed)

if true_state == :w_state
	const ρ_true = get_w_state(q) # true density matrix
elseif true_state == :ghz_state
	const ρ_true = get_ghz_state(q) # true density matrix
elseif true_state == :random_state
	const ρ_true = get_random_state(q) # true density matrix
else
	error("Unknown true_state: $true_state")
end
# const ρ_true = get_random_state(q) # true density matrix


## Problem setup
problem_setup_time = @elapsed begin
	# POVMs
	# const POVMs = get_pauli_povms(q) # Pauli POVMs
	const POVMs_positive = get_pauli_povms_positive(q)
	const idx_obs = rand(1:M, N)
	const outcomes = measure(ρ_true, POVMs_positive, idx_obs)
	const data = generate_data(POVMs_positive, idx_obs, outcomes) # if q = 7, it cause OOM
	const frequency, POVMs = generate_data_freq(POVMs_positive, idx_obs, outcomes)


	loss_func_ = (ρ::Hermitian{<:Complex}) -> @inline loss_func(ρ, data)
	gradient_ = (ρ::Hermitian{<:Complex}) -> @inline gradient(ρ, data)
	loss_and_gradient_ = (ρ::Hermitian{<:Complex}) -> @inline loss_and_gradient(ρ, data)
	loss_func_bm_ = (halved::Matrix{<:Complex}) -> @inline loss_func_bm(halved, data)
	gradient_bm_ = (halved::Matrix{<:Complex}) -> @inline gradient_bm(halved, data)
	loss_and_gradient_bm_ =
		(halved::Matrix{<:Complex}) -> @inline loss_and_gradient_bm(halved, data)

	# loss_func_ = (ρ::Hermitian{<:Complex}) -> @inline loss_func(ρ, frequency, POVMs)
	# gradient_ = (ρ::Hermitian{<:Complex}) -> @inline gradient(ρ, frequency, POVMs)
	# loss_and_gradient_ =
	# 	(ρ::Hermitian{<:Complex}) -> @inline loss_and_gradient(ρ, frequency, POVMs)
	# loss_func_bm_ =
	# 	(halved::Matrix{<:Complex}) -> @inline loss_func_bm(halved, frequency, POVMs)
	# gradient_bm_ =
	# 	(halved::Matrix{<:Complex}) -> @inline gradient_bm(halved, frequency, POVMs)
	# loss_and_gradient_bm_ =
	# 	(halved::Matrix{<:Complex}) ->
	# 		@inline loss_and_gradient_bm(halved, frequency, POVMs)
	# hessian_bm_ =
	# 	(halved::Matrix{<:Complex}, v::Matrix{<:Complex}) ->
	# 		@inline hessian_bm(halved, v, frequency, POVMs)
end
@printf("Problem setup time: %.6f seconds\n", problem_setup_time)