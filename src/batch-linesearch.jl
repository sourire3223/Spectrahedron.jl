using Format
using KrylovKit
using LinearAlgebra
using Printf
using TimerOutputs
include("./utils.jl")
include("./math-utils.jl")


const ArmijoParams = @NamedTuple{α0::Float64, r::Float64, τ::Float64}

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

function projected_gradient_descent(
	x_init::Hermitian{T},
	n_epoch::Integer,
	;
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	armijo_params::ArmijoParams,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "PGD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = @inline output_functions.init()
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = x_init

	α0, r, τ = armijo_params.α0, armijo_params.r, armijo_params.τ

	@inline output_functions.push!(output, 0.0, 0.0, f(ρ))
	stop_early = false
	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)
			# Armijo line search
			α = α0
			ρα = ρ - α * g
			ρα .= @inline euclidean_projection(ρα)
			fval_α = f(ρα)

			round = 0
			while τ * real(g ⋅ (ρα - ρ)) + fval < fval_α && (round += 1) <= 30
				α *= r
				ρα .= ρ - α * g
				ρα .= @inline euclidean_projection(ρα)
				fval_α = f(ρα)
			end
			if round <= 30 && fval_α < fval
				ρ = ρα
			else
				stop_early = true
			end
		end

		# update output after step
		if !stop_early
			time = TimerOutputs.time(to["iteration"]) * 1e-9
			@inline output_functions.push!(output, float(t), time, fval_α, ρ)
		else
			@printf(
				"Max round reached or insufficient descent, stopping early at epoch %d.\n",
				t - 1
			)
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

function feasible_direction_pgd(
	x_init::Hermitian{T},
	n_epoch::Integer,
	;
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	armijo_params::ArmijoParams,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "feasible direction PGD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = @inline output_functions.init()
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = x_init

	α0, r, τ = armijo_params.α0, armijo_params.r, armijo_params.τ

	@inline output_functions.push!(output, 0.0, 0.0, f(ρ))
	stop_early = false
	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)
			# Armijo line search
			α = α0
			ρα = ρ - α * g
			ρα .= @inline euclidean_projection(ρα)
			d = ρα - ρ
			fval_α = f(ρα)

			round = 0
			while τ * real(g ⋅ d) + fval < fval_α && (round += 1) <= 30
				d *= r
				ρα .= ρ + d
				fval_α = f(ρα)
			end
			if round <= 30 && fval_α < fval
				ρ = ρα
			else
				stop_early = true
			end
		end

		# update output after step
		if !stop_early
			time = TimerOutputs.time(to["iteration"]) * 1e-9
			@inline output_functions.push!(output, float(t), time, fval_α, ρ)
		else
			@printf(
				"Max round reached or insufficient descent, stopping early at epoch %d.\n",
				t - 1
			)
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


function emd_step(
	ρ::Hermitian{T},
	g::Hermitian{T},
	α::Float64,
	;
	atol::Float64 = 1e-12,
)::Hermitian{T} where {T<:Number}
	eig = eigen(Hermitian(ρ))
	mask = eig.values .> atol
	Λ = Diagonal(eig.values[mask])
	logΛ = log(Λ)
	U = eig.vectors[:, mask]

	g = Hermitian(U' * g * U)
	g .= Hermitian(logΛ - α * g)
	g -= opnorm(g) * I # for numerical stability softmax(x) = softmax(x - max(x))
	ρ_new = Hermitian(U * exp(g) * U')
	ρ_new /= tr(ρ_new)
	return ρ_new
end
function entropic_mirror_descent(
	x_init::Hermitian{T},
	n_epoch::Integer,
	;
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	armijo_params::ArmijoParams,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "EMD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = @inline output_functions.init()
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = x_init

	α0, r, τ = armijo_params.α0, armijo_params.r, armijo_params.τ
	@inline output_functions.push!(output, 0.0, 0.0, f(ρ), ρ)
	stop_early = false

	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)

			# Armijo line search
			α = α0
			ρα = @inline emd_step(ρ, g, α)
			fval_α = f(ρα)

			round = 0
			while τ * real(g ⋅ (ρα - ρ)) + fval < fval_α && (round += 1) <= 30
				α *= r
				ρα .= @inline emd_step(ρ, g, α)
				fval_α = f(ρα)
			end
			if round <= 30 && fval_α < fval
				ρ = ρα
			else
				stop_early = true
			end
		end

		# update output after step
		if !stop_early
			time = TimerOutputs.time(to["iteration"]) * 1e-9
			@inline output_functions.push!(output, float(t), time, fval_α, ρ)
		else
			@printf(
				"Max round reached or insufficient descent, stopping early at epoch %d.\n",
				t - 1
			)
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


function linear_oracle(g::Hermitian{T})::Hermitian{T} where {T<:Number}
	# solve min_{X∈S} tr(gX) where S={X≥0, tr(X)=1}
	_, vec, _ = eigsolve(g, 1, :SR; ishermitian = true)
	return Hermitian(vec[1] * vec[1]')
end


function frank_wolfe(
	x_init::Hermitian{T},
	n_epoch::Integer,
	;
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	armijo_params::ArmijoParams,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "FW"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = @inline output_functions.init()
	to = TimerOutput()
	reset_timer!()


	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = x_init

	α0, r, τ = armijo_params.α0, armijo_params.r, armijo_params.τ
	@inline output_functions.push!(output, 0.0, 0.0, f(ρ), ρ)
	stop_early = false

	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)
			# Armijo line search
			α = α0
			direction = @inline linear_oracle(g) - ρ
			ρα = ρ + α * direction
			fval_α = f(ρα)

			round = 0
			g_dir_inner = real(g ⋅ direction)
			while τ * α * g_dir_inner + fval < fval_α && (round += 1) <= 30
				α *= r
				ρα .= ρ + α * direction
				fval_α = f(ρα)
			end
			if round <= 30 && fval_α < fval
				ρ = ρα
			else
				stop_early = true
			end
		end

		# update output after step
		if !stop_early
			time = TimerOutputs.time(to["iteration"]) * 1e-9
			@inline output_functions.push!(output, float(t), time, fval_α, ρ)
		else
			@printf(
				"Max round reached or insufficient descent, stopping early at epoch %d.\n",
				t - 1
			)
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

function bw_rgd_helper(ρ::Hermitian{T}, g::Hermitian{T}) where {T<:Number}
	λ = g ⋅ ρ
	gI = (g - λ * I)
	gIρ = gI * ρ
	gIρgI = gIρ * gI
	gIρ_ρgI = gIρ + gIρ'

	# ρα = ρ - 2 * α * gIρ_ρgI + 4 * α^2 * gIρgI
	# ρα /= tr(ρα)
	return Hermitian(gIρ_ρgI), Hermitian(gIρgI)

end

function bw_rgd(
	x_init::Hermitian{T},
	n_epoch::Integer,
	;
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	armijo_params::ArmijoParams,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "BW-RGD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = @inline output_functions.init()
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = x_init

	α0, r, τ = armijo_params.α0, armijo_params.r, armijo_params.τ
	@inline output_functions.push!(output, 0.0, 0.0, f(ρ), ρ)
	stop_early = false

	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)

			# Armijo line search
			α = α0
			gρ_ρg, gρg = @inline bw_rgd_helper(ρ, g)
			ρα = Hermitian(ρ - 2 * α * gρ_ρg + 4 * α^2 * gρg)
			ρα /= tr(ρα)
			fval_α = f(ρα)

			round = 0
			rgrad_norm² = 4 * real(tr(gρg))
			while τ * α * -rgrad_norm² + fval < fval_α && (round += 1) <= 30
				α *= r
				ρα .= Hermitian(ρ - 2 * α * gρ_ρg + 4 * α^2 * gρg)
				ρα /= tr(ρα)
				fval_α = f(ρα)
			end
			if round <= 30 && fval_α < fval
				ρ = ρα
			else
				stop_early = true
			end
		end

		# update output after step
		if !stop_early
			time = TimerOutputs.time(to["iteration"]) * 1e-9
			@inline output_functions.push!(output, float(t), time, fval_α, ρ)
		else
			@printf(
				"Max round reached or insufficient descent, stopping early at epoch %d.\n",
				t - 1
			)
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


function sphere_tangent(x::Matrix{T}, v::Matrix{T}) where {T<:Number}
	# project v onto the tangent space at x on the unit sphere
	return v - real(x ⋅ v) * x
end


function sphere_rgd(
	x_init::Matrix{T},
	n_epoch::Integer,
	;
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	armijo_params::ArmijoParams,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Matrix{T},Dict{String,Any}} where {T<:Number}
	name = "Sphere-RGD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = @inline output_functions.init()
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	Y = x_init

	α0, r, τ = armijo_params.α0, armijo_params.r, armijo_params.τ
	@inline output_functions.push!(output, 0.0, 0.0, f(Y), Y)
	stop_early = false

	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(Y)

			# Armijo line search
			α = α0
			rie_grad = @inline sphere_tangent(Y, g)
			Yα = Y - α * rie_grad
			Yα /= norm(Yα)
			fval_α = f(Yα)

			round = 0
			rgrad_norm² = norm(rie_grad)^2
			while τ * α * -rgrad_norm² + fval < fval_α && (round += 1) <= 30
				α *= r
				Yα .= Y - α * rie_grad
				Yα /= norm(Yα)
				fval_α = f(Yα)
			end
			if round <= 30 && fval_α < fval
				Y = Yα
			else
				stop_early = true
			end
		end

		# update output after step
		if !stop_early
			time = TimerOutputs.time(to["iteration"]) * 1e-9
			@inline output_functions.push!(output, float(t), time, fval_α, Y)
		else
			@printf(
				"Max round reached or insufficient descent, stopping early at epoch %d.\n",
				t - 1
			)
			break
		end

	end
	time = TimerOutputs.time(to["iteration"]) * 1e-9
	@printf("Elapsed time: %.6f seconds\n", time)

	return Y, output
end