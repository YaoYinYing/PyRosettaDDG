import os
from pyrosetta_ddg.ddg_parallel import mutant_parser,timing, run_cart_ddg

pose_path =  "test/lowest_cart_relaxed_hg3.pdb"

mutants='87A,54C,78A_32T'

inputs_=mutant_parser(mutants_str=mutants)


print(inputs_)


with timing('Cartesian ddG'):
    results=run_cart_ddg(pdb_file=pose_path,mutants=inputs_, save_place='save_parallel-fix-parallel', nproc=os.cpu_count())
print(results)

