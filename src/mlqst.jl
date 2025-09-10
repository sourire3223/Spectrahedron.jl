using LinearAlgebra
using Kronecker
using Random
using Statistics
using Tullio



function calc_fidelity(ρ::Hermitian{<:Complex}, σ::Hermitian{<:Complex})::Real
	S = sqrt(σ)
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


function measure(
	state::Hermitian{<:Complex},
	povms_positive::Array{<:Complex,3},
	idx_obs::Vector{<:Integer},
)::Vector{Bool}
	ρ = state
	POVM = povms_positive
	n = length(idx_obs)

	prob = @tullio prob[k] := real(ρ ⋅ POVM[:, :, k])
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

function loss_func(state::Hermitian{<:Complex}, data::Array{<:Complex,3})::Real
	ρ = state
	prob = @tullio prob[k] := real(ρ ⋅ data[:, :, k])
	return mean(-log.(prob))

end

function gradient(
	state::Hermitian{<:Complex},
	data::Array{<:Complex,3},
)::Hermitian{<:Complex}
	ρ = state
	prob = @tullio prob[k] := real(ρ ⋅ data[:, :, k])
	grad = @tullio grad[i, j] := -data[i, j, k] / prob[k]
	return Hermitian(grad / size(data, 3))

end



function loss_and_gradient(
	state::Hermitian{<:Complex},
	data::Array{<:Complex,3},
)::Tuple{Real,Hermitian{<:Complex}}
	ρ = state
	prob = @tullio prob[k] := real(ρ ⋅ data[:, :, k])
	loss = mean(-log.(prob))
	grad = @tullio grad[i, j] := -data[i, j, k] / prob[k]
	return loss, Hermitian(grad / size(data, 3))
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

