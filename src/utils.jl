using Printf

const Output = Dict{String,Vector}
function init_output()::Output
	output = Dict()
	output["n_epoch"] = Float64[]
	output["fval"] = Float64[]
	output["elapsed_time"] = Float64[]
	return output
end

function push_output!(
	output::Output,
	n_epoch::Float64,
	time::Float64,
	fval::Float64,
	x_t::Any = nothing,
	;
	verbose::Bool = false,
)::Output
	push!(output["n_epoch"], n_epoch)
	push!(output["fval"], fval)
	push!(output["elapsed_time"], time)

	if verbose
		@printf("%.1f\t%.2e\t%.8e\n", n_epoch, time, fval)
	end
	return output
end

function pop_output!(output::Output)::Nothing
	pop!(output["n_epoch"])
	pop!(output["fval"])
	pop!(output["elapsed_time"])
end


function update_output!(
	output::Output,
	index::Integer,
	n_epoch::Float64,
	time::Float64,
	fval::Float64,
	x_t::Any = nothing,
	;
	verbose::Bool = false,
)
	output["n_epoch"][index] = n_epoch
	output["fval"][index] = fval
	output["elapsed_time"][index] = time

	if verbose
		@printf("%.1f\t%.2e\t%.8e\n", n_epoch, time, fval)
	end
end

function trim_output!(output::Output, len::Integer)
	for (_, v) in output
		resize!(v, len)
	end
end

function print_output(io::IO, output::Dict, index::Integer, verbose::Bool)
	@printf(
		io,
		"%.1f\t%.15e\t%.15e\n",
		output["n_epoch"][index],
		output["elapsed_time"][index],
		output["fval"][index]
	)
	if verbose
		@printf(
			"%.1f\t%.15e\t%.15e\n",
			output["n_epoch"][index],
			output["elapsed_time"][index],
			output["fval"][index]
		)
	end
	flush(io)
end

function write_output!(io::IO, output::Dict)
	for i in 1:length(output["n_epoch"])
		@printf(
			io,
			"%.1f\t%.15e\t%.15e\n",
			output["n_epoch"][i],
			output["elapsed_time"][i],
			output["fval"][i]
		)
	end
	flush(io)
end


const OutputFunctions = @NamedTuple{
	init::typeof(init_output),
	push!::typeof(push_output!),
	pop!::typeof(pop_output!),
}
const output_functions::OutputFunctions =
	(init = init_output, push! = push_output!, pop! = pop_output!)
