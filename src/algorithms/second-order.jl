using Printf
using TimerOutputs
using LinearAlgebra
include("../math-utils.jl")
include("../utils.jl")


function sphere_tangent(x::Matrix{T}, v::Matrix{T}) where {T<:Number}
	# project v onto the tangent space at x on the unit sphere
	return v - real(x ⋅ v) * x
end

function ehess2sphere_hess(
	x::Matrix{T},
	egrad::Matrix{T},
	ehess_v::Matrix{T},
	v::Matrix{T},
) where {T<:Number}
	# Riemannian hessian on the sphere manifold
	return sphere_tangent(x, ehess_v - real(x ⋅ egrad) * v)
end


function tCG(
	x::Matrix{T},
	egrad::Matrix{T},
	ehessian::Function,
	radius::Real;
	max_iter::Integer = 100,
) where {T<:Number}

	# Truncated Conjugate Gradient method for the trust-region subproblem
	# min 1/2 <η, H η> + <g, η>
	# s.t. ||η|| <= radius
	η₀ = zeros(T, size(egrad))
	r₀ = sphere_tangent(x, egrad)
	δ₀ = -r₀
	Δ = radius

	r₀_norm = norm(r₀)
	η = η₀
	r = r₀
	δ = δ₀
	r_r = real(r ⋅ r)


	Hᵉδ = Matrix{T}(undef, size(x))
	Hʳδ = Matrix{T}(undef, size(x))

	for _ in 1:max_iter
		Hᵉδ .= @inline ehessian(x, δ)
		Hʳδ .= @inline ehess2sphere_hess(x, egrad, Hᵉδ, δ)
		δ_Hʳδ = real(δ ⋅ Hʳδ)

		if δ_Hʳδ <= 0
			# negative curvature, solve ||η + τδ|| = Δ
			# τ^2||δ||^2 + 2τ(η⋅δ) + ||η||^2 - Δ^2 = 0
			a = real(δ ⋅ δ)
			b = 2 * real(η ⋅ δ)
			c = real(η ⋅ η) - Δ^2
			τ = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
			η .= η + τ * δ
			return η
		end


		α = r_r / δ_Hʳδ
		η .= η + α * δ
		if norm(η) >= Δ
			# solve ||η + τδ|| = Δ
			# τ^2||δ||^2 + 2τ(η⋅δ) + ||η||^2 - Δ^2 = 0
			η .= η - α * δ # revert the last step
			a = real(δ ⋅ δ)
			b = 2 * real(η ⋅ δ)
			c = real(η ⋅ η) - Δ^2
			τ = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
			η .= η + τ * δ
			return η
		end

		r .= r + α * Hʳδ
		r⁺_r⁺ = real(r ⋅ r)
		if sqrt(r⁺_r⁺) <= r₀_norm * min(0.1, r₀_norm)
			return η
		end

		β = r⁺_r⁺ / r_r
		δ .= -r + β * δ
		δ .= @inline sphere_tangent(x, δ)
		r_r = r⁺_r⁺
	end
	return η
end
const TrustRegionParams = @NamedTuple{
	Δ̄::Float64, # initial step size for line search
	Δ₀::Float64,  # initial trust region radius
	ρ′::Float64, # step size reduction factor for line search
}
function trust_region_tCG(
	x_init::Matrix{T},
	n_epoch::Integer,
	;
	loss_func::Function,
	gradient::Function,
	hessian::Function,
	loss_and_gradient::Function = x -> (loss_func(x), gradient(x)),
	trust_region_params::TrustRegionParams = (Δ̄ = 10.0, Δ₀ = 5.0, ρ′ = 0.0),
	output_functions::OutputFunctions = output_functions,
)::Tuple{Matrix{T},Dict{String,Any}} where {T<:Number}
	name = "trust_region_tCG"
	println(name * " starts.")
	# @printf(io, "%s\n%d\n%d\n", name, n_epoch, n_rate)
	output = output_functions.init()
	to = TimerOutput()
	reset_timer!()

	f = loss_func
	∇f = gradient
	∇²f = hessian
	f_and_∇f = loss_and_gradient # assume faster

	Y = x_init
	Δ̄, Δ₀, ρ′ = trust_region_params

	Δ = Δ₀
	@inline output_functions.push!(output, 0.0, 0.0, f(Y), Hermitian(Y * Y'))
	@inbounds for t in 1:n_epoch
		@timeit to "iteration" begin
			fval, g = f_and_∇f(Y)
			gʳ = sphere_tangent(Y, g)

			η = tCG(Y, g, ∇²f, Δ)
			Y_η = Y + η
			Y_η /= norm(Y_η)
			fval_η = f(Y_η)

			Hᵉη = ∇²f(Y, η)
			Hʳη = @inline ehess2sphere_hess(Y, g, Hᵉη, η)
			ρ = (fval - fval_η) / (-real(gʳ ⋅ η) - 0.5 * real(η ⋅ Hʳη))

			if ρ < 0.25
				Δ = 0.25 * Δ
			elseif ρ > 0.75 && abs(norm(η) - Δ) < 1e-8
				Δ = min(2 * Δ, Δ̄)
			else
				Δ = Δ
			end

			if ρ > ρ′
				Y = Y_η
			else
				Y = Y
			end
		end

		time = TimerOutputs.time(to["iteration"]) * 1e-9
		fval = ρ > ρ′ ? fval_η : fval
		@inline output_functions.push!(output, float(t), time, fval, Hermitian(Y * Y'))
		if norm(gʳ) < 1e-8
			break
		end
	end
	time = TimerOutputs.time(to["iteration"]) * 1e-9
	@printf("Elapsed time: %.6f seconds\n", time)

	return Y, output
end