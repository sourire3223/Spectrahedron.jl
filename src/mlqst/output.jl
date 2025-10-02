include("../utils.jl")
include("./functions.jl")

function init_output_mlqst()::Output
	output = Dict()
	output["n_epoch"] = Float64[]
	output["fval"] = Float64[]
	output["elapsed_time"] = Float64[]
	output["fidelity"] = Float64[]
	return output
end

function push_output_mlqst!(
	output::Output,
	n_epoch::Float64,
	time::Float64,
	fval::Float64,
	x_t::Any = nothing,
	;
	verbose::Bool = true,
)::Output
	push!(output["n_epoch"], n_epoch)
	push!(output["fval"], fval)
	push!(output["elapsed_time"], time)
	push!(output["fidelity"], @inline calc_fidelity(ρ_true, x_t))
	if verbose
		@printf(
			"%.1f (%.2fs): loss=%.8e, fidelity=%.8e\n",
			n_epoch,
			time,
			fval,
			output["fidelity"][end]
		)
	end
	return output
end

const output_functions_mlqst::OutputFunctions =
	(init = init_output_mlqst, push! = push_output_mlqst!, pop! = pop_output!)