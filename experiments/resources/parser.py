
def register_configurations(parser):

    # Outer Optimization (Loss Discovery) Hyper-parameters
    parser.add_argument("--population_size", required=False, type=int)
    parser.add_argument("--num_generations", required=False, type=int)
    parser.add_argument("--crossover_rate", required=False, type=float)
    parser.add_argument("--mutation_rate", required=False, type=float)
    parser.add_argument("--elitism_rate", required=False, type=float)
    parser.add_argument("--tournament_size", required=False, type=int)
    parser.add_argument("--max_height", required=False, type=int)
    parser.add_argument("--init_min_height", required=False, type=int)
    parser.add_argument("--init_max_height", required=False, type=int)

    # Inner Optimization (Loss Optimization) Hyper-parameters
    parser.add_argument("--meta_gradient_steps", required=False, type=int)
    parser.add_argument("--meta_learning_rate", required=False, type=float)
    parser.add_argument("--inner_gradient_steps", required=False, type=int)
    parser.add_argument("--base_gradient_steps", required=False, type=int)
    parser.add_argument("--base_learning_rate", required=False, type=float)
    parser.add_argument("--base_weight_decay", required=False, type=float)
    parser.add_argument("--base_momentum", required=False, type=float)
    parser.add_argument("--base_nesterov", required=False, type=bool)
    parser.add_argument("--base_batch_size", required=False, type=int)

    # Loss Function Learning Filter Hyper-parameters
    parser.add_argument("--filter_gradient_steps", required=False, type=int)
    parser.add_argument("--filter_sample_size", required=False, type=int)

    # Loss Function Learning Objectives
    parser.add_argument("--outer_objective_name", required=False, type=str)
    parser.add_argument("--inner_objective_name", required=False, type=str)

    # Meta Testing Hyper-parameters
    parser.add_argument("--testing_gradient_steps", required=False, type=int)
    parser.add_argument("--testing_learning_rate", required=False, type=float)
    parser.add_argument("--testing_weight_decay", required=False, type=float)
    parser.add_argument("--testing_momentum", required=False, type=float)
    parser.add_argument("--testing_nesterov", required=False, type=bool)
    parser.add_argument("--testing_batch_size", required=False, type=int)
    parser.add_argument("--testing_milestones", required=False, type=int, nargs="+")
    parser.add_argument("--testing_gamma", required=False, type=float)

    # Experiment Settings
    parser.add_argument("--output_path", required=False, type=str)
    parser.add_argument("--verbose", required=False, type=int)
