using LinearAlgebra
using Printf
using TimerOutputs
using KrylovKit
include("../utils.jl")
include("../math-utils.jl")

function RρR(
	ρ_init::Hermitian{T},
	n_epoch::Integer,
	;
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "RρR"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = @inline output_functions.init()
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = copy(ρ_init)

	@inline output_functions.push!(output, 0.0, 0.0, f(ρ), ρ)
	stop_early = false

	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			R = -∇f(ρ)
			ρ .= Hermitian(R * ρ * R)
			ρ /= tr(ρ)
		end

		time = TimerOutputs.time(to["iteration"]) * 1e-9
		@inline output_functions.push!(output, float(t), time, f(ρ), ρ)
		if opnorm(R - I) < 1e-8
			@printf("Converged at epoch %d\n", t)
			break
		end
	end
	time = TimerOutputs.time(to["iteration"]) * 1e-9
	@printf("Elapsed time: %.6f seconds\n", time)
	if !is_density_matrix(ρ)
		@warn "The output is not a density matrix."
	end

	return ρ, output
end
function qem_step!(
	ρ::Hermitian{T},
	g::Hermitian{T},
	;
	atol::Float64 = 1e-12,
)::Hermitian{T} where {T<:Number}
	eig = eigen(Hermitian(ρ))
	mask = eig.values .> atol
	Λ = Diagonal(eig.values[mask])
	logΛ = log(Λ)
	U = eig.vectors[:, mask]

	g = Hermitian(U' * g * U)
	g .= Hermitian(logΛ + log(-g))
	ρ .= Hermitian(U * exp(g) * U')
	ρ /= tr(ρ)
	return ρ
end
function QEM_last(
	ρ_init::Hermitian{T},
	n_epoch::Integer,
	;
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "QEM"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = @inline output_functions.init()
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = copy(ρ_init)

	@inline output_functions.push!(output, 0.0, 0.0, f(ρ), ρ)

	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			g = ∇f(ρ)
			@inline qem_step!(ρ, g)
		end

		time = TimerOutputs.time(to["iteration"]) * 1e-9
		@inline output_functions.push!(output, float(t), time, f(ρ), ρ)
		if opnorm(g + I) < 1e-8
			@printf("Converged at epoch %d\n", t)
			break
		end
	end
	time = TimerOutputs.time(to["iteration"]) * 1e-9
	@printf("Elapsed time: %.6f seconds\n", time)
	if !is_density_matrix(ρ)
		@warn "The output is not a density matrix. Tr(ρ)=$(tr(ρ))"
	end

	return ρ, output
end
