using LinearAlgebra

function is_density_matrix(ρ::AbstractMatrix{T}, eps::Real = 1e-12)::Bool where {T<:Number}
	# Hermitian check
	ishermitian(ρ) || return false

	# Positive semidefinite check (allowing numerical noise)
	isposdef(ρ + eps * I) || return false

	# Trace normalization
	return isapprox(real(tr(ρ)), 1.0; atol = eps, rtol = 0)
end
