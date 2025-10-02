using LinearAlgebra
using Printf
using TimerOutputs
include("../utils.jl")
include("../math-utils.jl")


function simplex_projection(y::Vector{T}, tol::Float64 = 1e-12)::Vector{T} where {T<:Real}
	# compute argmin_{y∈Δ} ||x-y||_2 where Δ={y∈R^d: y≥0, ∑_i y_i=1}
	# Wang and Carreira-Perpiñán. 2013. "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application".
	d = length(y)
	u = sort(y; rev = true)
	cumsum_u = cumsum(u)
	ρ = findlast(i -> u[i] + (1 - cumsum_u[i]) / i > tol, 1:d)
	λ = (1 - cumsum_u[ρ]) / ρ
	return max.(y .+ λ, 0)
end

function euclidean_projection(X::Hermitian{<:Number})::Hermitian{<:Number}
	eig = eigen(X)
	new_eigvals = @inline simplex_projection(eig.values)
	return Hermitian(eig.vectors * Diagonal(new_eigvals) * eig.vectors')
end
function pgd_polyak_stepsize(
	x_init::Hermitian{T},
	n_epoch::Integer,
	;
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	f_min::Real = 0.0,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "PGD with Polyak stepsize"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = @inline output_functions.init()
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ::Hermitian{T} = x_init

	@inline output_functions.push!(output, 0.0, 0.0, f(ρ), ρ)
	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)
			α = (fval - f_min) / (norm(g)^2 + eps())
			println("Step size at epoch $t: $α")
			ρ .= ρ - α * g
			ρ .= @inline euclidean_projection(ρ)
		end

		# update output after step
		time = TimerOutputs.time(to["iteration"]) * 1e-9
		@inline output_functions.push!(output, float(t), time, f(ρ), ρ)
	end
	time = TimerOutputs.time(to["iteration"]) * 1e-9
	@printf("Elapsed time: %.6f seconds\n", time)
	if !is_density_matrix(ρ)
		@warn "The output is not a density matrix."
	end
	return ρ, output
end