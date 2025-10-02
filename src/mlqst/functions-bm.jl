using LinearAlgebra
include("./functions.jl")
# throughout the code, ρ = halved * halved'

function get_random_state_halved(
	n_qubits::Integer,
	rank::Integer = 2^n_qubits,
)::Matrix{<:Complex}
	d = 2^n_qubits
	halved = randn(ComplexF64, d, rank)
	halved /= norm(halved)
	return halved
end


function get_w_state_halved(n_qubits::Integer)::Matrix{<:Complex}
	w = zeros(ComplexF64, 2^n_qubits)
	for i ∈ 1:n_qubits
		w[2^(i-1)+1] = 1
	end
	w /= norm(w)
	return w
end


function loss_func_bm(
	halved::Matrix{T},
	data::AbstractArray{T,3},
)::real(T) where {T<:Complex}
	ρ = Hermitian(halved * halved')
	return @inline loss_func(ρ, data)
end

function loss_func_bm(
	halved::Matrix{Complex{T}},
	frequency::Vector{T},
	data::AbstractArray{Complex{T},3},
)::T where {T<:AbstractFloat}
	ρ = Hermitian(halved * halved')
	return @inline loss_func(ρ, frequency, data)
end


function gradient_bm(
	halved::Matrix{T},
	data::AbstractArray{T,3},
)::Matrix{T} where {T<:Complex}
	ρ = Hermitian(halved * halved')
	g = @inline gradient(ρ, data)
	return g * halved
end

function gradient_bm(
	halved::Matrix{Complex{T}},
	frequency::Vector{T},
	data::AbstractArray{Complex{T},3},
)::Hermitian{Complex{T}} where {T<:AbstractFloat}
	ρ = Hermitian(halved * halved')
	g = @inline gradient(ρ, frequency, data)
	return g * halved
end

function loss_and_gradient_bm(
	halved::Matrix{T},
	data::AbstractArray{T,3},
)::Tuple{real(T),Matrix{T}} where {T<:Complex}
	ρ = Hermitian(halved * halved')
	loss, grad = @inline loss_and_gradient(ρ, data)
	return loss, grad * halved
end


function loss_and_gradient_bm(
	halved::Matrix{Complex{T}},
	frequency::Vector{T},
	data::AbstractArray{Complex{T},3},
)::Tuple{T,Matrix{Complex{T}}} where {T<:AbstractFloat}
	ρ = Hermitian(halved * halved')
	loss, grad = @inline loss_and_gradient(ρ, frequency, data)
	return loss, grad * halved
end

function hessian_bm(
	halved::Matrix{T},
	v::Matrix{T},
	data::AbstractArray{T,3},
)::Matrix{T} where {T<:Complex}
	ρ = Hermitian(halved * halved')
	return @inline hessian(ρ, Hermitian(v * halved' + halved * v'), data) * halved +
				   @inline gradient(ρ, data) * v
end

function hessian_bm(
	halved::Matrix{Complex{T}},
	v::Matrix{Complex{T}},
	frequency::Vector{T},
	data::AbstractArray{Complex{T},3},
)::Matrix{Complex{T}} where {T<:AbstractFloat}
	ρ = Hermitian(halved * halved')
	return @inline hessian(ρ, Hermitian(v * halved' + halved * v'), frequency, data) *
				   halved + @inline gradient(ρ, frequency, data) * v
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
