from pyrosetta_ddg.ddg_parallel import DDG_Runner, mutant_parser,timing

pose_path =  "test/lowest_cart_relaxed_hg3.pdb"

mutants='87A,54C,78A_32T'

inputs_=mutant_parser(mutants_str=mutants)

print(inputs_)

runner=DDG_Runner(pdb_input=pose_path)
with timing('Cartesian ddG'):
    results=runner.run_cart_ddg(mutants=inputs_)

print(results)

