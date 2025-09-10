using Format
using KrylovKit
using LinearAlgebra
using Printf
using TimerOutputs
using Tullio
include("./utils.jl")


function simplex_projection(y::Vector{<:Real}, tol::Real = 1e-12)::Vector{<:Real}
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
	new_eigvals = simplex_projection(eig.values)
	return Hermitian(eig.vectors * Diagonal(new_eigvals) * eig.vectors')
end

function PGD(
	x_init::Hermitian{T},
	n_epoch::Integer,
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Union{Function,Nothing} = nothing,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "PGD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = init_output(n_epoch)
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	if loss_and_gradient !== nothing
		f_and_∇f = loss_and_gradient # assume faster
	else
		f_and_∇f = (x) -> (f(x), ∇f(x))
	end

	ρ = x_init

	α0::Float64 = 10
	r::Float64 = 0.5
	τ::Float64 = 0.5

	@inbounds for t ∈ 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)
			# Armijo line search
			α = α0
			ρα = ρ - α * g

			ρα = euclidean_projection(ρα)
			round = 0
			while τ * real(g ⋅ (ρα - ρ)) + fval < f(ρα) && round < 30
				α *= r
				ρα .= ρ - α * g
				ρα = euclidean_projection(ρα)
				round += 1
			end
			if round < 30
				ρ = ρα
			else
				trim_output!(output, t-1)
				break
			end
		end

		update_output!(output, t, t, fval, TimerOutputs.time(to["iteration"]) * 1e-9)
		# print_output(io, output, t, VERBOSE)
	end

	return ρ, output
end
function EMD(
	x_init::Hermitian{T},
	n_epoch::Integer,
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Union{Function,Nothing} = nothing,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "EMD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = init_output(n_epoch)
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	if loss_and_gradient !== nothing
		f_and_∇f = loss_and_gradient # assume faster
	else
		f_and_∇f = (x) -> (f(x), ∇f(x))
	end

	ρ = x_init

	α0::Float64 = 10
	r::Float64 = 0.5
	τ::Float64 = 0.5

	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)

			# Armijo line search
			α = α0
			ρα = exp(Hermitian(log(ρ)) - α * g) # prevent wrong return type, https://github.com/JuliaLang/LinearAlgebra.jl/blob/98723dff52d8a3003822cb43e28910fbde9c73f0/src/symmetric.jl#L950C1-L959C4
			ρα /= tr(ρα)

			round = 0
			while τ * real(g ⋅ (ρα - ρ)) + fval < f(ρα) && round < 30
				α *= r
				ρα .= exp(Hermitian(log(ρ)) - α * g) # prevent wrong return type, https://github.com/JuliaLang/LinearAlgebra.jl/blob/98723dff52d8a3003822cb43e28910fbde9c73f0/src/symmetric.jl#L950C1-L959C4
				ρα /= tr(ρα)
				round += 1
			end
			if round < 30
				ρ = ρα
			else
				trim_output!(output, t-1)
				break
			end
		end

		update_output!(output, t, t, fval, TimerOutputs.time(to["iteration"]) * 1e-9)
		# print_output(io, output, t, VERBOSE)
	end

	return ρ, output
end


function linear_oracle(g::Hermitian{T})::Hermitian{T} where {T<:Number}
	# solve min_{X∈S} tr(gX) where S={X≥0, tr(X)=1}
	_, vec, _ = eigsolve(g, 1, :SR; ishermitian = true)
	return Hermitian(vec[1] * vec[1]')
end


function FW(
	x_init::Hermitian{T},
	n_epoch::Integer,
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Union{Function,Nothing} = nothing,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "FW"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = init_output(n_epoch)
	to = TimerOutput()
	reset_timer!()


	f = loss_func
	∇f = gradient
	if loss_and_gradient !== nothing
		f_and_∇f = loss_and_gradient # assume faster
	else
		f_and_∇f = (x) -> (f(x), ∇f(x))
	end

	ρ = x_init

	α0::Float64 = 1 - 1e-12
	r::Float64 = 0.8
	τ::Float64 = 0.5

	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)
			# Armijo line search
			α = α0
			direction = linear_oracle(g) - ρ
			ρα = ρ + α * direction
			round = 0

			g_dir = real(g ⋅ direction)
			while τ * α * g_dir + fval < f(ρα) && round < 30
				# while τ * real(g ⋅ (α*direction)) + fval < f(ρα) && round < 10
				α *= r
				ρα .= ρ + α * direction
				round += 1
			end
			if round < 30
				ρ = ρα
			else
				trim_output!(output, t-1)
				break
			end
		end

		update_output!(output, t, t, fval, TimerOutputs.time(to["iteration"]) * 1e-9)
		# print_output(io, output, t, VERBOSE)
	end

	return ρ, output
end

function bwrgd_helper(ρ::Hermitian{T}, g::Hermitian{T}) where {T<:Number}
	λ = g ⋅ ρ
	gI = (g - λ * I)
	gIρ = gI * ρ
	gIρgI = gIρ * gI
	gIρ_ρgI = gIρ + gIρ'

	# ρα = ρ - 2 * α * gIρ_ρgI + 4 * α^2 * gIρgI
	# ρα /= tr(ρα)
	return Hermitian(gIρ_ρgI), Hermitian(gIρgI)

end

function BWRGD(
	x_init::Hermitian{T},
	n_epoch::Integer,
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Union{Function,Nothing} = nothing,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "BW-RGD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = init_output(n_epoch)
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	if loss_and_gradient !== nothing
		f_and_∇f = loss_and_gradient # assume faster
	else
		f_and_∇f = (x) -> (f(x), ∇f(x))
	end

	ρ = x_init

	α0::Float64 = 1e2
	r::Float64 = 0.5
	τ::Float64 = 0.1

	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)

			# Armijo line search
			α = α0
			gρ_ρg, gρg = bwrgd_helper(ρ, g)
			ρα = Hermitian(ρ - 2 * α * gρ_ρg + 4 * α^2 * gρg)
			ρα /= tr(ρα)


			round = 0
			rie_grad_norm2 = 8 * real(tr(gρg))
			while τ * α * -rie_grad_norm2 + fval < f(ρα) && round < 30
				# while τ * real(g ⋅ (ρα - ρ)) + fval < f(ρα) && round < 10
				α *= r
				ρα .= Hermitian(ρ - 2 * α * gρ_ρg + 4 * α^2 * gρg)
				ρα /= tr(ρα)
				round += 1
			end
			if round < 30
				ρ = ρα
			else
				trim_output!(output, t-1)
				break
			end
		end

		update_output!(output, t, t, fval, TimerOutputs.time(to["iteration"]) * 1e-9)
		# print_output(io, output, t, VERBOSE)
	end

	return ρ, output
end