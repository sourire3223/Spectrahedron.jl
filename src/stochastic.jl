using LinearAlgebra
using Printf
using TimerOutputs
using Random
include("./utils.jl")
include("./math-utils.jl")

function sphere_tangent(x::Matrix{T}, v::Matrix{T}) where {T<:Number}
	# project v onto the tangent space at x on the unit sphere
	return v - real(x ⋅ v) * x
end

function sphere_srgd(
	x_init::Matrix{T},
	n_epochs::Integer,
	;
	data::Array{T,3},
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	stepsize::Real = 1e2,
	output_functions::OutputFunctions = output_functions,
)::Tuple{Matrix{T},Dict{String,Any}} where {T<:Number}
	name = "Sphere-RGD"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)

	n_data = size(data, 3)
	max_iter = n_epochs * n_data
	period = n_data

	output = output_functions.init()
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	Y = x_init


	@inline output_functions.push!(output, 0.0, 0.0, f(Y, data), Y)
	rand_indices = rand(1:(n_data÷100), max_iter)  # Pre-generate random indices for stochastic updates
	@inbounds for (t, i) in enumerate(rand_indices)
		@timeit to "iteration" begin
			g = @inline ∇f(Y, view(data,:,:,((i*100-99):(i*100))))
			rie_grad = @inline sphere_tangent(Y, g)
			Y .= Y - stepsize / t * rie_grad
			Y .= Y / norm(Y)
		end

		# update output after step
		if t % period == 0 || t == max_iter
			time = TimerOutputs.time(to["iteration"]) * 1e-9
			fval = f(Y, data)
			output_functions.push!(output, float(t) / n_data, time, fval, Y)
		end

	end
	time = TimerOutputs.time(to["iteration"]) * 1e-9
	@printf("Elapsed time: %.6f seconds\n", time)

	return Y, output
end
