using Printf


function init_output(len::Integer)
	output = Dict()
	output["n_epoch"] = zeros(Float64, len)
	output["fval"] = zeros(Float64, len)
	output["elapsed_time"] = zeros(Float64, len)
	return output
end


function update_output!(output, index, t, fval, time)
	output["n_epoch"][index] = t
	output["fval"][index] = fval
	output["elapsed_time"][index] = time
end


function print_output(io, output, index, verbose)
	@printf(
		io,
		"%.1f\t%E\t%E\t%E\n",
		output["n_epoch"][index],
		output["elapsed_time"][index],
		output["fval"][index]
	)
	if verbose
		@printf(
			"%.1f\t%E\t%E\t%E\n",
			output["n_epoch"][index],
			output["fidelity"][index],
			output["elapsed_time"][index],
			output["fval"][index]
		)
	end
	flush(io)
end
