using LinearAlgebra
using Printf
using TimerOutputs
using Random
include("../utils.jl")
include("../math-utils.jl")


function q_soft_bayes(
	x_init::Hermitian{T},
	n_epochs::Integer,
	;
	data::Array{T,3},
	loss_func::Function,
	gradient::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	output_functions::OutputFunctions = output_functions,
)::Tuple{Hermitian{T},Dict{String,Any}} where {T<:Number}
	name = "Q-Soft-Bayes"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)

	d, ~, n_data = size(data)
	max_iter = n_epochs * n_data
	period = n_data

	output = output_functions.init()
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	f_and_∇f = loss_and_gradient # assume faster

	ρ = copy(x_init)

	η::Float64 = sqrt(log(d) / max_iter / d)
	η = η / (1.0 + η)
	σ = (1.0 - η) * Hermitian(Matrix{ComplexF64}(I, d, d))
	@inline output_functions.push!(output, 0.0, 0.0, f(ρ, data), ρ)
	rand_indices = rand(1:n_data, max_iter)  # Pre-generate random indices for stochastic updates
	@inbounds for (t, i) in enumerate(rand_indices)
		@timeit to "iteration" begin
			g = @inline ∇f(ρ, view(data,:,:,[i]))
			ρ .= exp(Hermitian(log(ρ) + log(σ - η * g)))
			ρ.data ./= tr(ρ)
		end

		# update output after step
		if t % period == 0 || t == max_iter
			time = TimerOutputs.time(to["iteration"]) * 1e-9
			fval = f(ρ, data)
			output_functions.push!(output, float(t) / n_data, time, fval, ρ)
		end

	end
	time = TimerOutputs.time(to["iteration"]) * 1e-9
	@printf("Elapsed time: %.6f seconds\n", time)
	println(to)
	return ρ, output
end
