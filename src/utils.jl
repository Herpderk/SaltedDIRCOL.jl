const DiffFloat = Union{AbstractFloat, FD.Dual}

"""
"""
function get_module_function_names(
    mod::Module
)::Vector{Symbol}
    syms = names(mod, all=true)
    functions = filter(sym::Symbol -> isa(getproperty(mod,sym), Function), syms)
    return functions
end

"""
"""
function get_module_function(
    mod::Module,
    func_name::Symbol
)::Function
    # Get all functions in the module
    names = get_module_function_names(mod)
    # Find the appropriate function or throw an error
    for name in names
        if name == func_name
            return getproperty(mod, func_name)
        end
    end
    error("The given function is not implemented in this module!")
end
