Base.@kwdef struct SARSOPSolver{LOW,UP} <: Solver
    epsilon::Float64    = 0.5
    precision::Float64  = 1e-3
    kappa::Float64      = 0.5
    delta::Float64      = 1e-1
    max_time::Float64   = 1.0
    max_steps::Int      = typemax(Int)
    verbose::Bool       = true
    init_lower::LOW     = BlindLowerBound(bel_res = 1e-2)
    init_upper::UP      = FastInformedBound(bel_res=1e-2)
    prunethresh::Float64= 0.10
end

function POMDPs.solve(solver::SARSOPSolver, pomdp::POMDP)
    tree = SARSOPTree(solver, pomdp)

    t0 = time()
    iterations = 0
    while time()-t0 < solver.max_time && root_diff(tree) > solver.precision
        sample!(solver, tree)
        backup!(tree)
        prune!(solver, tree)
        iterations += 1
    end
    return AlphaVectorPolicy(
        pomdp,
        getproperty.(tree.Γ, :alpha),
        ordered_actions(pomdp)[getproperty.(tree.Γ, :action)]
    )
end

function track_bounds(tree::SARSOPTree, lower_bounds::Vector{Float64}, upper_bounds::Vector{Float64})
    """
    Store the values of lower and upper bounds (at the root of the tree) in vectors
    """
    push!(lower_bounds, tree.V_lower[1])
    push!(upper_bounds, tree.V_upper[1])
end

function solve_applied(solver::SARSOPSolver, pomdp::POMDP)
    println("ENTER modified 'solve' method")
    tree = SARSOPTree(solver, pomdp)

    lower_bounds::Vector{Float64} = []
    upper_bounds::Vector{Float64} = []
    track_bounds(tree, lower_bounds, upper_bounds)

    t0 = time()
    iterations = 0
    while time()-t0 < solver.max_time && root_diff(tree) > solver.precision
        sample!(solver, tree)
        backup!(tree)
        prune!(solver, tree)
        track_bounds(tree, lower_bounds, upper_bounds)
        iterations += 1
    end
    policy::AlphaVectorPolicy = AlphaVectorPolicy(
        pomdp,
        getproperty.(tree.Γ, :alpha),
        ordered_actions(pomdp)[getproperty.(tree.Γ, :action)]
    )
    return policy, lower_bounds, upper_bounds
end