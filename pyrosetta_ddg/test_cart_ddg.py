import multiprocessing
import os

import numpy as np
from pyrosetta_ddg.cart_ddg import DDGPayload, Mutant,create_score_function,pose_from_pdb, initialize_pyrosetta, mutant_parser, mutate_repack_func4, setup_ddg_payload, timing,deep_copy


def cart_ddg(p:DDGPayload)-> 'Mutant':
    
    scores = []

    # clone() is a shadow copy not a real deep copy. 
    # deep_copy() is also clone()
    # https://graylab.jhu.edu/PyRosetta.documentation/pyrosetta.rosetta.core.pose.html#pyrosetta.rosetta.core.pose.Pose.detached_copy
    newpose=deep_copy(pose)

    mutant=p.mutant
    
    for i in range(p.iterations):
        scorefxn = create_score_function("ref2015_cart")
        newpose = mutate_repack_func4(newpose,p.mutant, 6, scorefxn,verbose = False, cartesian = True, save_pose_to=p.save_pose_to_pdb(i))
        news = scorefxn(newpose)
        print(f'{mutant.id}.{i}: {news}')
        scores.append(news)

    mutant.scores=scores
    print(f'{str(mutant)}')
    return mutant

initialize_pyrosetta()

pose_path =  "test/1ubq.pdb"
pose = pose_from_pdb(pose_path)


all_as = sorted(list(set(pose.sequence())))
sites = np.arange(1,len(pose.sequence()) + 1)

print(f'{all_as=}')

mutants='57A,54C,76A_32T,32C,27F'

inputs_mutants=tuple([m for m in mutant_parser(mutants_str=mutants)])
inputs_=setup_ddg_payload(mutants=inputs_mutants, repeat_times=3, save_to='save_ddg_deep_copy_pose')

print(len(inputs_))
cores = os.cpu_count()
print(cores)


with timing('Cartesian ddG'):
    with multiprocessing.Pool(processes=cores) as pool:
        results = pool.starmap(cart_ddg,inputs_)

