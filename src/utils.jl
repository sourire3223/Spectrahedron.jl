using Printf


function init_output(len::Integer)
	output = Dict()
	output["n_epoch"] = zeros(Float64, len)
	output["fval"] = zeros(Float64, len)
	output["elapsed_time"] = zeros(Float64, len)
	return output
end


function update_output!(
	output::Dict,
	index::Integer,
	t::Float64,
	fval::Float64,
	time::Float64,
)
	output["n_epoch"][index] = t
	output["fval"][index] = fval
	output["elapsed_time"][index] = time
end

function trim_output!(output::Dict, len::Integer)
	for (_, v) in output
		resize!(v, len)
	end
end

function print_output(io::IO, output::Dict, index::Integer, verbose::Bool)
	@printf(
		io,
		"%.1f\t%E\t%E\n",
		output["n_epoch"][index],
		output["elapsed_time"][index],
		output["fval"][index]
	)
	if verbose
		@printf(
			"%.1f\t%E\t%E\n",
			output["n_epoch"][index],
			output["elapsed_time"][index],
			output["fval"][index]
		)
	end
	flush(io)
end

function write_output(io::IO, output::Dict)
	for i in 1:length(output["n_epoch"])
		@printf(
			io,
			"%.1f\t%E\t%E\n",
			output["n_epoch"][i],
			output["elapsed_time"][i],
			output["fval"][i]
		)
	end
	flush(io)
end
