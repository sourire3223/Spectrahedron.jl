using Format
using KrylovKit
using LinearAlgebra
using Printf
using TimerOutputs
include("./utils.jl")
include("./math_utils.jl")


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
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x));
	armijo_params::ArmijoParams,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "PGD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = output_functions.init(n_epoch + 1)
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = x_init

	α0, r, τ = armijo_params.α0, armijo_params.r, armijo_params.τ

	output_functions.update!(output, 1, 0.0, 0.0, f(ρ))
	stop_early = false
	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)
			# Armijo line search
			α = α0
			ρα = ρ - α * g
			ρα = @inline euclidean_projection(ρα)
			fval_α = f(ρα)

			round = 0
			while τ * real(g ⋅ (ρα - ρ)) + fval < fval_α && (round += 1) <= 30
				α *= r
				ρα .= ρ - α * g
				ρα = @inline euclidean_projection(ρα)
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
			output_functions.update!(output, t + 1, float(t), time, fval_α)
		else
			output_functions.trim!(output, t)
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
function entropic_mirror_descent(
	x_init::Hermitian{T},
	n_epoch::Integer,
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	;
	armijo_params::ArmijoParams,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "EMD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = output_functions.init(n_epoch + 1)
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = x_init

	α0, r, τ = armijo_params.α0, armijo_params.r, armijo_params.τ
	output_functions.update!(output, 1, 0.0, 0.0, f(ρ))
	stop_early = false

	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(ρ)

			# Armijo line search
			α = α0
			ρα = exp(Hermitian(log(ρ)) - α * g) # prevent wrong return type, https://github.com/JuliaLang/LinearAlgebra.jl/blob/98723dff52d8a3003822cb43e28910fbde9c73f0/src/symmetric.jl#L950C1-L959C4
			ρα /= tr(ρα)
			fval_α = f(ρα)

			round = 0
			while τ * real(g ⋅ (ρα - ρ)) + fval < fval_α && (round += 1) <= 30
				α *= r
				ρα .= exp(Hermitian(log(ρ)) - α * g) # prevent wrong return type, https://github.com/JuliaLang/LinearAlgebra.jl/blob/98723dff52d8a3003822cb43e28910fbde9c73f0/src/symmetric.jl#L950C1-L959C4
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
			output_functions.update!(output, t + 1, float(t), time, fval_α)
		else
			output_functions.trim!(output, t)
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
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	;
	armijo_params::ArmijoParams,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "FW"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = output_functions.init(n_epoch + 1)
	to = TimerOutput()
	reset_timer!()


	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = x_init

	α0, r, τ = armijo_params.α0, armijo_params.r, armijo_params.τ
	output_functions.update!(output, 1, 0.0, 0.0, f(ρ))
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
			output_functions.update!(output, t + 1, float(t), time, fval_α)
		else
			output_functions.trim!(output, t)
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
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	;
	armijo_params::ArmijoParams,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "BW-RGD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = output_functions.init(n_epoch + 1)
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = x_init

	α0, r, τ = armijo_params.α0, armijo_params.r, armijo_params.τ
	output_functions.update!(output, 1, 0.0, 0.0, f(ρ))
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
			rie_grad_norm2 = 4 * real(tr(gρg))
			while τ * α * -rie_grad_norm2 + fval < fval_α && (round += 1) <= 30
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
			output_functions.update!(output, t + 1, float(t), time, fval_α)
		else
			output_functions.trim!(output, t)
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


function rie_grad_sphere(halved::Matrix{T}, g::Matrix{T}) where {T<:Number}
	# Riemannian gradient on the sphere manifold
	return g - real(halved ⋅ g) * halved
end

function sphere_rgd(
	halved_init::Matrix{T},
	n_epoch::Integer,
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	;
	armijo_params::ArmijoParams,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Matrix{T},Dict{String,Any}} where {T<:Number}
	name = "Sphere-RGD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = output_functions.init(n_epoch + 1)
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	Y = halved_init

	α0, r, τ = armijo_params.α0, armijo_params.r, armijo_params.τ
	output_functions.update!(output, 1, 0.0, 0.0, f(Y))
	stop_early = false

	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(Y)

			# Armijo line search
			α = α0
			rie_grad = @inline rie_grad_sphere(Y, g)
			Yα = Y - α * rie_grad
			Yα /= norm(Yα)
			fval_α = f(Yα)

			round = 0
			rie_grad_norm2 = norm(rie_grad)^2
			while τ * α * -rie_grad_norm2 + fval < fval_α && (round += 1) <= 30
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
			output_functions.update!(output, t + 1, float(t), time, fval_α)
		else
			output_functions.trim!(output, t)
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