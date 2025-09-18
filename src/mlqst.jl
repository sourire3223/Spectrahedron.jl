using LinearAlgebra
using Kronecker
using Random
using Statistics
using Tullio


function calc_fidelity(ρ::Hermitian{T}, σ::Hermitian{T})::real(T) where {T<:Complex}
	S::Hermitian{T} = sqrt(σ)
	return real((tr(sqrt(S * ρ * S')))^2)
end


function get_random_state(n_qubits::Integer)::Hermitian{<:Complex}
	d = 2^n_qubits
	halved = randn(ComplexF64, d, d)
	ρ = halved * halved'
	ρ = ρ / tr(ρ)
	return Hermitian(ρ)
end


function get_w_state(n_qubits::Integer)::Hermitian{<:Complex}
	w = zeros(ComplexF64, 2^n_qubits)
	for i ∈ 1:n_qubits
		w[2^(i-1)+1] = 1
	end
	W = w * w'
	W = W / tr(W)
	return Hermitian(W)
end


function get_pauli_observables(n_qubits::Integer)::Array{ComplexF64,3}
	# Pauli matrices
	#TODO
	σ = Array{ComplexF64,3}(undef, 2, 2, 4)
	σ[:, :, 1] .= [0 1; 1 0]
	σ[:, :, 2] .= [0 -im; im 0]
	σ[:, :, 3] .= [1 0; 0 -1]
	σ[:, :, 4] .= [1 0; 0 1]

	d = 2^n_qubits    # dimension
	observables = Array{ComplexF64,3}(undef, d, d, 4^n_qubits)
	@inbounds for (i, indices) in
				  enumerate(Iterators.product(ntuple(_ -> 1:4, n_qubits)...))
		one = ones(ComplexF64, 1, 1)
		X = reduce(kron, (view(σ,:,:,j) for j in indices); init = one)
		observables[:, :, i] .= collect(X)
	end

	return observables
end


function get_pauli_povms_positive(n_qubits::Integer)::Array{ComplexF64,3}
	pauli_observables = get_pauli_observables(n_qubits)
	d = 2^n_qubits
	I_d = Matrix{ComplexF64}(I, d, d)
	POVM = (pauli_observables .+ I_d) ./ 2 # M_+ = (I + P)/2
	return POVM
end

function compute_prob(
	state::Hermitian{T},
	data::Array{T,3},
)::Vector{real(T)} where {T<:Complex}
	ρ = state
	d, ~, n = size(data)

	data_v = reshape(data, d*d, n)
	ρ_v = vec(Matrix(ρ))
	res = real.(ρ_v' * data_v)
	return vec(res)
end

function measure(
	state::Hermitian{<:Complex},
	povms_positive::Array{<:Complex,3},
	idx_obs::Vector{<:Integer},
)::Vector{Bool}
	ρ = state
	POVM = povms_positive
	n = length(idx_obs)

	prob = @inline compute_prob(ρ, POVM)
	rand_vals = rand(n)
	outcomes = [rand_vals[i] < prob[idx_obs[i]] for i in 1:n]

	return outcomes

end



function generate_data(
	povms_positive::Array{ComplexF64,3},
	idx_obs::Vector{Int},
	outcomes::Vector{Bool},
)::Array{ComplexF64,3}
	d = size(povms_positive, 1)
	n = length(idx_obs)

	data = Array{ComplexF64,3}(undef, d, d, n)
	I_d = Matrix{ComplexF64}(I, d, d)
	@inbounds for i in 1:n
		obs = view(povms_positive,:,:,idx_obs[i])
		data[:, :, i] .= outcomes[i] ? obs : I_d .- obs
	end
	return data
end


function loss_func(state::Hermitian{T}, data::Array{T,3}) where {T<:Complex}
	prob = @inline compute_prob(state, data)
	return mean(-log.(prob))
end

function gradient(state::Hermitian{T}, data::Array{T,3})::Hermitian{T} where {T<:Complex}
	prob = @inline compute_prob(state, data)

	d, ~, n = size(data)
	datav = reshape(data, d^2, n)
	weights = -1.0 ./ prob
	grad = similar(data, d^2)
	mul!(grad, datav, weights)
	return Hermitian(reshape(grad, d, d) / n)
end



function loss_and_gradient(
	state::Hermitian{T},
	data::Array{T,3},
)::Tuple{real(T),Hermitian{T}} where {T<:Complex}
	prob = @inline compute_prob(state, data)

	d, ~, n = size(data)
	datav = reshape(data, d^2, n)
	weights = -1.0 ./ prob
	grad = similar(state, d^2)
	mul!(grad, datav, weights)
	return mean(-log.(prob)), Hermitian(reshape(grad, d, d) / n)
end

# function log_barrier_projection(
# 	u::Array{Float64, 1},
# 	ε::Float64,
# )
# 	# compute argmin_{x∈Δ} D_h(x,u) where h(x)=∑_{i=1}^d -log(x_i)
# 	# minimize ϕ(θ) = θ - ∑_i log(θ + u_i^{-1})

# 	θ::Float64 = 1 - minimum(1 ./ u)
# 	a::Array{Float64, 1} = 1 ./ ((1 ./ u) .+ θ)
# 	∇::Float64 = 1 - sum(a)
# 	∇2::Float64 = a ⋅ a
# 	λt::Float64 = abs(∇) / sn_qubitsrt(∇2)

# 	while λt > ε
# 		a = 1 ./ ((1 ./ u) .+ θ)
# 		∇ = 1 - norm(a, 1)
# 		∇2 = a ⋅ a
# 		θ = θ - ∇ / ∇2
# 		λt = abs(∇) / sn_qubitsrt(∇2)
# 	end

# 	return (1 ./ ((1 ./ u) .+ θ))
# end


# function α(ρ, v)
# 	return -real(tr(ρ * v * ρ) / tr(ρ * ρ))
# end


# function dual_norm2(ρ, σ)
# 	A = ρ * σ
# 	return real(tr(A * A))
# end
