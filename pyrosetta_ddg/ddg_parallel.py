# modified from https://github.com/ccbiozhaw/FitLan/blob/main/rosetta_ddG_calculation/ddg.ipynb

import contextlib
import multiprocessing
import os
import time
import random
from typing import Optional, Union
from dataclasses import dataclass


from pyrosetta import pose_from_file,create_score_function

from  pyrosetta.rosetta.core.pack.task import *
from  pyrosetta.rosetta.core.select import *
from pyrosetta.rosetta import core, protocols,basic
from pyrosetta.rosetta.core.pose import Pose

from pyrosetta import init
from pyrosetta.io import pose_from_pdb


#Python



#Core Includes
from  pyrosetta.rosetta.core.kinematics import MoveMap
from  pyrosetta.rosetta.core.kinematics import FoldTree
from  pyrosetta.rosetta.core.pack.task import TaskFactory
from  pyrosetta.rosetta.core.pack.task import operation
from  pyrosetta.rosetta.core.simple_metrics import metrics
from  pyrosetta.rosetta.core.select.residue_selector import (
    ResidueIndexSelector, NotResidueSelector,
    NeighborhoodResidueSelector, PrimarySequenceNeighborhoodSelector,
    OrResidueSelector
)
from  pyrosetta.rosetta.core import select
from  pyrosetta.rosetta.core.select.movemap import *

#Protocol Includes
from pyrosetta.rosetta.protocols import minimization_packing as pack_min
from pyrosetta.rosetta.protocols import relax as rel
from pyrosetta.rosetta.protocols.antibody.residue_selector import CDRResidueSelector
from pyrosetta.rosetta.protocols.antibody import *
from pyrosetta.rosetta.protocols.loops import *

from pyrosetta.toolbox import *

def initialize_pyrosetta():
    # Initialize PyRosetta once per process with unique random seed
    seed = random.randint(1, 1000000)
    init_options = f"-default_max_cycles 200 -missing_density_to_jump -ex1 -ex2aro -ignore_zero_occupancy false -fa_max_dis 9 -mute all -constant_seed -jran {seed}"
    init(init_options)



@dataclass
class Mutation:
    pos: int
    aa: str

    def __post_init__(self):
        if isinstance(self.pos,str) and self.pos.isdigit():
            self.pos = int(self.pos)

    def __str__(self):
        return f"{self.pos}{self.aa}"
        
    
@dataclass
class Mutant:
    mutations: list[Mutation]
    scores: tuple[float]=None

    @property
    def id(self)-> str:
        return "_".join(str(m) for m in self.mutations) if self.mutations else 'WT'


    def squash(self, mutants: list['Mutant'], override: bool=False) -> 'Mutant':
        if len(set(m.id for m in mutants)) >1:
            raise ValueError("Mutant must have at least one unique id")

        if self.scores and not override:
            raise ValueError(f'This mutant has scores: {self.scores}, use override=True to override')

        m=self.copy
        m.scores = tuple(x for m in mutants for x in m.scores)

        return m



    @property
    def astuple(self) -> tuple[Union[int, str]]:
        '''
        Returns a tuple of (pos[int], aa[str])
        '''
        return tuple((m.pos, m.aa,) for m in self.mutations)


    @property
    def copy(self):
        return Mutant(mutations=self.mutations, scores=self.scores)
    
    @property
    def all_pos(self) -> list[int]:
        return [m.pos for m in self.mutations]
    

    def __str__(self):
        return self.id + f', Score: {self.scores if self.scores else ""}' 


    @staticmethod
    def from_str(mut_str: str) -> 'Mutant':
        return Mutant(
            mutations=[
                Mutation(
                    pos=int(''.join(filter(str.isdigit, m))), 
                    aa=''.join(filter(str.isalpha, m)),
                    ) for m in mut_str.split('_')] 
            )


@contextlib.contextmanager
def timing(msg: str):
    print(f'Started {msg}')
    tic = time.time()
    yield
    toc = time.time()
    print(f'Finished {msg} in {toc - tic:3f} seconds')


def mutate_repack_func4(pose, mutant: Mutant, repack_radius, sfxn, ddg_bbnbrs=1, verbose=False, cartesian=True, max_iter=None, save_pose_to: Optional[str]='save', iteration: Optional[int]=0) -> Pose:
    if cartesian:
        sfxn.set_weight(core.scoring.ScoreTypeManager.score_type_from_name('cart_bonded'), 0.5)
        #sfxn.set_weight(atom_pair_constraint, 1)#0.5
        sfxn.set_weight(core.scoring.ScoreTypeManager.score_type_from_name('pro_close'), 0)
        #logger.warning(pyrosetta.rosetta.basic.options.get_boolean_option('ex1'))#set_boolean_option( '-ex1', True )
        #pyrosetta.rosetta.basic.options.set_boolean_option( 'ex2', True )
        
    if save_pose_to and iteration is not None:
        os.makedirs(save_pose_to, exist_ok=True)
    
    # early return if there exists a pdb file with such a mutant.
    if os.path.isfile((expected_pdb_save:=os.path.join(save_pose_to, f'{mutant.id}_i.{iteration}.pdb'))):
        if verbose:
            print(f"Loading pose from saved {expected_pdb_save}")
        return pose_from_pdb(expected_pdb_save)

    
    from  pyrosetta.rosetta.core.pack.task import operation
    
    #Cloning of the pose including all settings
    working_pose = pose.clone()


    # Create residue selectors for mutations
    mutant_selectors = []
    for position in mutant.all_pos:
        mutant_selector = ResidueIndexSelector(str(position))
        mutant_selectors.append(mutant_selector)

    # Combine mutant selectors
    if len(mutant_selectors) == 1:
        combined_mutant_selector = mutant_selectors[0]
    else:
        combined_mutant_selector = OrResidueSelector()
        for sel in mutant_selectors:
            combined_mutant_selector.add_residue_selector(sel)
    

    #Select all except mutant
    all_nand_mutant_selector = core.select.residue_selector.NotResidueSelector()
    all_nand_mutant_selector.set_residue_selector(combined_mutant_selector)

    # Select neighbors within repack_radius
    nbr_or_mutant_selector = NeighborhoodResidueSelector()
    nbr_or_mutant_selector.set_focus_selector(combined_mutant_selector)
    nbr_or_mutant_selector.set_distance(repack_radius)
    nbr_or_mutant_selector.set_include_focus_in_subset(True)

    #Select mutant and it's sequence neighbors
    seq_nbr_or_mutant_selector = core.select.residue_selector.PrimarySequenceNeighborhoodSelector(ddg_bbnbrs, ddg_bbnbrs, combined_mutant_selector, False)            

    #Select mutant, it's seq neighbors and it's surrounding neighbors
    seq_nbr_or_nbr_or_mutant_selector = core.select.residue_selector.OrResidueSelector()
    seq_nbr_or_nbr_or_mutant_selector.add_residue_selector(seq_nbr_or_mutant_selector)
    seq_nbr_or_nbr_or_mutant_selector.add_residue_selector(nbr_or_mutant_selector)    

    if verbose:
        print(f'mutant_selector: {core.select.residue_selector.selection_positions(combined_mutant_selector.apply(working_pose))}')
        print(f'all_nand_mutant_selector: {core.select.residue_selector.selection_positions(all_nand_mutant_selector.apply(working_pose))}')
        print(f'nbr_or_mutant_selector: {core.select.residue_selector.selection_positions(nbr_or_mutant_selector.apply(working_pose))}')
        print(f'seq_nbr_or_mutant_selector: {core.select.residue_selector.selection_positions(seq_nbr_or_mutant_selector.apply(working_pose))}')
        print(f'seq_nbr_or_nbr_or_mutant_selector: {core.select.residue_selector.selection_positions(seq_nbr_or_nbr_or_mutant_selector.apply(working_pose))}')
    
        

    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.IncludeCurrent())

    #Set all residues except mutant to false for design and repacking
    prevent_repacking_rlt = operation.PreventRepackingRLT()
    prevent_subset_repacking = operation.OperateOnResidueSubset(prevent_repacking_rlt, all_nand_mutant_selector, False )
    tf.push_back(prevent_subset_repacking)


    # Specify mutations
    for pos, aa in mutant.astuple:
        resfile_cmd = f"PIKAA {aa}"
        resfile_comm = protocols.task_operations.ResfileCommandOperation(ResidueIndexSelector(str(pos)), resfile_cmd)
        tf.push_back(resfile_comm)


    #Apply packing of rotamers of mutant
    packer =protocols.minimization_packing.PackRotamersMover()
    packer.score_function(sfxn)
    packer.task_factory(tf)
    if verbose:
        print(tf.create_task_and_apply_taskoperations(working_pose))
    packer.apply(working_pose)
        
    #allow the movement for bb for the mutant + seq. neighbors, and sc for neigbor in range, seq. neighbor and mutant
    movemap = core.select.movemap.MoveMapFactory()
    movemap.all_jumps(False)
    movemap.add_bb_action(core.select.movemap.mm_enable, seq_nbr_or_mutant_selector)
    movemap.add_chi_action(core.select.movemap.mm_enable, seq_nbr_or_nbr_or_mutant_selector)
    
    #for checking if all has been selected correctly
    #if verbose:
    mm  = movemap.create_movemap_from_pose(working_pose)
    

    #Generate a TaskFactory
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.IncludeCurrent())
    #tf.push_back(operation.NoRepackDisulfides())

    #prevent all residues except selected from design and repacking
    prevent_repacking_rlt = operation.PreventRepackingRLT()
    prevent_subset_repacking = operation.OperateOnResidueSubset(prevent_repacking_rlt, seq_nbr_or_nbr_or_mutant_selector, True )
    tf.push_back(prevent_subset_repacking)

    # allow selected residues only repacking (=switch off design)
    restrict_repacking_rlt = operation.RestrictToRepackingRLT()
    restrict_subset_repacking = operation.OperateOnResidueSubset(restrict_repacking_rlt , seq_nbr_or_nbr_or_mutant_selector, False)
    tf.push_back(restrict_subset_repacking)


    #Perform a FastRelax
    fastrelax =protocols.relax.FastRelax()
    fastrelax.set_scorefxn(sfxn)
    
    if cartesian:
        fastrelax.cartesian(True)
    if max_iter:
        fastrelax.max_iter(max_iter)
        
    fastrelax.set_task_factory(tf)
    fastrelax.set_movemap_factory(movemap)
    fastrelax.set_movemap_disables_packing_of_fixed_chi_positions(True)
    
    if verbose:
        print(tf.create_task_and_apply_taskoperations(working_pose))
    fastrelax.apply(working_pose)

    if save_pose_to and iteration is not None:
        os.makedirs(save_pose_to, exist_ok=True)
        working_pose.dump_pdb(expected_pdb_save)
    return working_pose

def setup_ddg_payload(pdb_fp: str,mutants: list[Mutant], repeat_times:int=3,save_to: str='save')-> list[tuple[Mutant, int]]:
    payload=[(pdb_fp, m, iteration, save_to) for m in mutants for iteration in range(repeat_times)]
    print(f'Payload Number: {len(payload)}')
    return payload


def cart_ddg(pdb_file:str, mutant: Mutant, iteration: int=0, save_place: str='save') -> Mutant:

    # init(f"-default_max_cycles 200 -missing_density_to_jump -ex1 -ex2aro -ignore_zero_occupancy false -fa_max_dis 9 -mute all -seed_offset {iteration}")
    
    newpose = pdb2pose(pdb_file)
    scorefxn = create_score_function("ref2015_cart")
    print(f'Mutate: {str(mutant)}: Iter. {iteration}')
    newpose = mutate_repack_func4(newpose, mutant, 6, scorefxn, verbose = False, cartesian = True, save_pose_to=save_place, iteration=iteration)
    news = scorefxn(newpose)
    mutant_copy=mutant.copy
    mutant_copy.scores=[news]
    return mutant_copy

def run_cart_ddg(pdb_file, mutants: list[Mutant], save_place, nproc: int=os.cpu_count()) ->list[Mutant]:
    
        
    print(f'CPU in uses: {nproc}')

    payload=setup_ddg_payload(pdb_fp=pdb_file, repeat_times=3, mutants=mutants, save_to=save_place)
    

    with multiprocessing.Pool(processes=nproc, initializer=initialize_pyrosetta) as pool:
        results = pool.starmap(cart_ddg,payload)

    # # loky backend fails on init of pyrosetta
    # results: list[Mutant] = Parallel(n_jobs=self.nproc, backend='multiprocessing', return_as='list', verbose=51)(
    #             delayed(self.cart_ddg)(**m)
    #             for m in payload
    #         )

    print(results)
        
    return [m.squash(mutants=[r for r in results if r.id==m.id]) for m in mutants]
 

    
def file2pose(filepath: str) -> Pose:
    return pose_from_file(filepath)

        
        

def pdb2pose(pdb_file_path: str) -> Pose:
    return pose_from_pdb(pdb_file_path)


def mutant_parser(mutants_str: str, use_wt:bool=True) -> list[Mutant]:
    '''
    Construct a list of Mutant object, comma-separated
    '''
    if use_wt:
        lm: list[Mutant]=[Mutant([])]
    else:
        lm: list[Mutant]=[]
    lm.extend([Mutant.from_str(m_str) for m_str in mutants_str.split(',') if m_str])
    return lm

