import contextlib
from dataclasses import dataclass, asdict, field
import multiprocessing
import random
import time
from typing import Callable, Optional, Union
import warnings
import pyrosetta.io
from pyrosetta.rosetta.core.pack.task import *
from pyrosetta.rosetta.protocols import *
from pyrosetta.rosetta.core.select import *

# Python
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.teaching import *

from pyrosetta import create_score_function, pose_from_pdb

# Core Includes
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.kinematics import FoldTree
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation
from pyrosetta.rosetta.core.simple_metrics import metrics
from pyrosetta.rosetta.core.select import residue_selector as selections
from pyrosetta.rosetta.core import select
from pyrosetta.rosetta.core.select.movemap import *

# Protocol Includes
from pyrosetta.rosetta.protocols import minimization_packing as pack_min
from pyrosetta.rosetta.protocols import relax as rel
from pyrosetta.rosetta.protocols.antibody.residue_selector import (
    CDRResidueSelector,
)
from pyrosetta.rosetta.protocols.antibody import *
from pyrosetta.rosetta.protocols.loops import *
from pyrosetta.rosetta.protocols.relax import FastRelax
import numpy as np
import pandas as pd

from pyrosetta.toolbox import *

DDG_INIT_FLAGS = "-default_max_cycles 200 -missing_density_to_jump -ex1 -ex2aro -ignore_zero_occupancy false -fa_max_dis 9 -mute all"

init(DDG_INIT_FLAGS)


# a full deep copy of the src pose obj.
def deep_copy(pose: Pose) -> Pose:
    p = Pose()
    p.detached_copy(pose)
    return p


@contextlib.contextmanager
def timing(msg: str):
    print(f'Started {msg}')
    tic = time.time()
    yield
    toc = time.time()
    print(f'Finished {msg} in {toc - tic:3f} seconds')


@dataclass
class Mutation:
    pos: int
    aa: str

    def __post_init__(self):
        if isinstance(self.pos, str) and self.pos.isdigit():
            self.pos = int(self.pos)

    def __str__(self):
        return f"{self.pos}{self.aa}"


@dataclass
class Mutant:
    mutations: list[Mutation]

    saved_pose: dict[int, str] = field(default_factory=dict)
    saved_score: dict[int, float] = field(default_factory=dict)

    pdb_path: Optional[tuple[str]] = None

    @property
    def scores(self) -> tuple[float]:
        return tuple(self.saved_score.values())

    @scores.setter
    def scores(self, scores: tuple[float]):
        self.saved_score = {i: v for i, v in enumerate(scores)}

    @property
    def id(self) -> str:
        return (
            "_".join(str(m) for m in self.mutations)
            if self.mutations
            else 'WT'
        )

    def squash(
        self, mutants: list['Mutant'], override: bool = False
    ) -> 'Mutant':
        if len(set(m.id for m in mutants)) > 1:
            raise ValueError("Mutant must have at least one unique id")

        if self.scores and not override:
            raise ValueError(
                f'This mutant has scores: {self.scores}, use override=True to override'
            )

        m = self.copy
        m.scores = tuple(x for m in mutants for x in m.scores)

        return m

    @property
    def mean_score(self) -> float:
        return np.mean(self.scores)

    @property
    def std_score(self) -> float:
        return np.std(self.scores)

    @property
    def astuple(self) -> tuple[Union[int, str]]:
        '''
        Returns a tuple of (pos[int], aa[str])
        '''
        return tuple(
            (
                m.pos,
                m.aa,
            )
            for m in self.mutations
        )

    @property
    def copy(self):
        return Mutant(**asdict(self))

    @property
    def all_pos(self) -> list[int]:
        return [m.pos for m in self.mutations]

    def __str__(self):
        score_str = (
            f', Score: {self.scores}, mean: {self.mean_score}, std: {self.std_score}'
            if self.scores
            else ""
        )
        return self.id + score_str

    @staticmethod
    def from_str(mut_str: str) -> 'Mutant':
        return Mutant(
            mutations=[
                Mutation(
                    pos=int(''.join(filter(str.isdigit, m))),
                    aa=''.join(filter(str.isalpha, m)),
                )
                for m in mut_str.split('_')
            ]
        )

    @property
    def asdataframe(self) -> pd.DataFrame:
        if not self.scores:
            raise ValueError("This mutant has no scores")
        d = {'id': self.id}

        # scores
        d.update({f'iter_{i}': s for i, s in enumerate(self.scores)})

        # summary
        d.update({'mean': self.mean_score, 'std': self.std_score})
        return pd.DataFrame(d, index=[0])


@dataclass
class DDGPayload:
    mutant: Mutant
    save_poses_to_pdb: Union[list[str], None]

    @property
    def iterations(self) -> int:
        return len(self.save_poses_to_pdb)

    def save_pose_to_pdb(self, i: int) -> Union[str, None]:
        return self.save_poses_to_pdb[i] if self.save_poses_to_pdb else None


def mutant_parser(mutants_str: str, use_wt: bool = True) -> list[Mutant]:
    '''
    Construct a list of Mutant object, comma-separated
    '''
    if use_wt:
        lm: list[Mutant] = [Mutant([])]
    else:
        lm: list[Mutant] = []
    lm.extend(
        [Mutant.from_str(m_str) for m_str in mutants_str.split(',') if m_str]
    )
    return lm


def setup_ddg_payload(
    mutants: list[Mutant], repeat_times: int = 3, save_to: Optional[str] = None
) -> list[DDGPayload]:
    payload = [
        (
            DDGPayload(
                m,
                [
                    (
                        os.path.join(save_to, f'{m.id}.i.{i}.pdb')
                        if save_to is not None
                        else None
                    )
                    for i in range(repeat_times)
                ],
            ),
        )
        for m in mutants
    ]
    print(f'Payload Number: {len(payload)}')
    return payload


# https://forum.rosettacommons.org/node/11126
def mutate_repack_func4(
    pose,
    mutant: Mutant,
    repack_radius,
    sfxn,
    ddg_bbnbrs=1,
    verbose=False,
    cartesian=True,
    max_iter=None,
    save_pose_to: Optional[str] = None,
):
    from pyrosetta.rosetta.core.pack.task import operation

    # logger.warning("Interface mode not implemented (should be added!)")

    if cartesian:
        sfxn.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreTypeManager.score_type_from_name(
                'cart_bonded'
            ),
            0.5,
        )
        # sfxn.set_weight(atom_pair_constraint, 1)#0.5
        sfxn.set_weight(
            pyrosetta.rosetta.core.scoring.ScoreTypeManager.score_type_from_name(
                'pro_close'
            ),
            0,
        )
        # logger.warning(pyrosetta.rosetta.basic.options.get_boolean_option('ex1'))#set_boolean_option( '-ex1', True )
        # pyrosetta.rosetta.basic.options.set_boolean_option( 'ex2', True )

    if save_pose_to and os.path.isfile(save_pose_to):
        print(f'Recover saved pose from pdb: {save_pose_to}')
        return pose_from_pdb(save_pose_to)

    # Cloning of the pose including all settings
    working_pose = pose.clone()

    # Create residue selectors for mutations
    mutant_selectors = []
    for position in mutant.all_pos:
        mutant_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
            str(position)
        )
        mutant_selectors.append(mutant_selector)

    # Combine mutant selectors
    if len(mutant_selectors) == 1:
        combined_mutant_selector = mutant_selectors[0]
    else:
        combined_mutant_selector = (
            pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector()
        )
        for sel in mutant_selectors:
            combined_mutant_selector.add_residue_selector(sel)

    # Select all except mutant
    all_nand_mutant_selector = (
        pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector()
    )
    all_nand_mutant_selector.set_residue_selector(combined_mutant_selector)

    # Select neighbors with mutant
    nbr_or_mutant_selector = (
        pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    )
    nbr_or_mutant_selector.set_focus_selector(combined_mutant_selector)
    nbr_or_mutant_selector.set_distance(repack_radius)
    nbr_or_mutant_selector.set_include_focus_in_subset(True)

    # Select mutant and it's sequence neighbors
    seq_nbr_or_mutant_selector = pyrosetta.rosetta.core.select.residue_selector.PrimarySequenceNeighborhoodSelector(
        ddg_bbnbrs, ddg_bbnbrs, combined_mutant_selector, False
    )

    # Select mutant, it's seq neighbors and it's surrounding neighbors
    seq_nbr_or_nbr_or_mutant_selector = (
        pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector()
    )
    seq_nbr_or_nbr_or_mutant_selector.add_residue_selector(
        seq_nbr_or_mutant_selector
    )
    seq_nbr_or_nbr_or_mutant_selector.add_residue_selector(
        nbr_or_mutant_selector
    )

    if verbose:
        print(
            f'mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(combined_mutant_selector.apply(working_pose))}'
        )
        print(
            f'all_nand_mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(all_nand_mutant_selector.apply(working_pose))}'
        )
        print(
            f'nbr_or_mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(nbr_or_mutant_selector.apply(working_pose))}'
        )
        print(
            f'seq_nbr_or_mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(seq_nbr_or_mutant_selector.apply(working_pose))}'
        )
        print(
            f'seq_nbr_or_nbr_or_mutant_selector: {pyrosetta.rosetta.core.select.residue_selector.selection_positions(seq_nbr_or_nbr_or_mutant_selector.apply(working_pose))}'
        )

    # Mutate residue and pack rotamers before relax
    # if list(pose.sequence())[target_position-1] != mutant:
    # generate packer task
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.IncludeCurrent())

    # Set all residues except mutant to false for design and repacking
    prevent_repacking_rlt = operation.PreventRepackingRLT()
    prevent_subset_repacking = operation.OperateOnResidueSubset(
        prevent_repacking_rlt, all_nand_mutant_selector, False
    )
    tf.push_back(prevent_subset_repacking)

    # Assign mutant residue to be designed and repacked
    # Specify mutations
    for pos, aa in mutant.astuple:
        resfile_cmd = f"PIKAA {aa}"
        resfile_comm = pyrosetta.rosetta.protocols.task_operations.ResfileCommandOperation(
            pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(
                str(pos)
            ),
            resfile_cmd,
        )
        tf.push_back(resfile_comm)

    # Apply packing of rotamers of mutant
    packer = (
        pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
    )
    packer.score_function(sfxn)
    packer.task_factory(tf)
    if verbose:
        logger.warning(tf.create_task_and_apply_taskoperations(working_pose))
    packer.apply(working_pose)

    # allow the movement for bb for the mutant + seq. neighbors, and sc for neigbor in range, seq. neighbor and mutant
    movemap = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
    movemap.all_jumps(False)
    movemap.add_bb_action(
        pyrosetta.rosetta.core.select.movemap.mm_enable,
        seq_nbr_or_mutant_selector,
    )
    movemap.add_chi_action(
        pyrosetta.rosetta.core.select.movemap.mm_enable,
        seq_nbr_or_nbr_or_mutant_selector,
    )

    # for checking if all has been selected correctly
    # if verbose:
    mm = movemap.create_movemap_from_pose(working_pose)

    logger.info(mm)

    # Generate a TaskFactory
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.IncludeCurrent())
    # tf.push_back(operation.NoRepackDisulfides())

    # prevent all residues except selected from design and repacking
    prevent_repacking_rlt = operation.PreventRepackingRLT()
    prevent_subset_repacking = operation.OperateOnResidueSubset(
        prevent_repacking_rlt, seq_nbr_or_nbr_or_mutant_selector, True
    )
    tf.push_back(prevent_subset_repacking)

    # allow selected residues only repacking (=switch off design)
    restrict_repacking_rlt = operation.RestrictToRepackingRLT()
    restrict_subset_repacking = operation.OperateOnResidueSubset(
        restrict_repacking_rlt, seq_nbr_or_nbr_or_mutant_selector, False
    )
    tf.push_back(restrict_subset_repacking)

    # Perform a FastRelax
    fastrelax = pyrosetta.rosetta.protocols.relax.FastRelax()
    fastrelax.set_scorefxn(sfxn)

    if cartesian:
        fastrelax.cartesian(True)
    if max_iter:
        fastrelax.max_iter(max_iter)

    fastrelax.set_task_factory(tf)
    fastrelax.set_movemap_factory(movemap)
    fastrelax.set_movemap_disables_packing_of_fixed_chi_positions(True)

    if verbose:
        logger.info(tf.create_task_and_apply_taskoperations(working_pose))
    fastrelax.apply(working_pose)

    if save_pose_to:
        os.makedirs(os.path.dirname(save_pose_to), exist_ok=True)
        working_pose.dump_pdb(save_pose_to)
    return working_pose


# def reseed() -> int:

#     options = pyrosetta.rosetta.basic.options.process()
#     rgs = pyrosetta.rosetta.basic.random.RandomGeneratorSettings()
#     old_seed = pyrosetta.rosetta.basic.random.determine_random_number_seed(rgs)
#     rgs.initialize_from_options(options)
#     new_seed = random.randint(0, 0xFFFFFF)
#     pyrosetta.rosetta.basic.random.init_random_generators(
#         new_seed, rgs.rng_type()
#     )
#     print(f'Seed: {old_seed} -> {new_seed}')
#     return new_seed


@dataclass
class ddGRunner:

    pose_path: str

    save_to: str = 'save'
    repeat_times: int = 3
    nproc: int = os.cpu_count()
    repack_radius: int = 6

    verbose: bool = False
    relax_max_iter: Optional[int] = None

    def __post_init__(self):
        if 'serialization' in pyrosetta.version():
            warnings.warn(
                RuntimeWarning(
                    'ddGRunner does not require serialization build of PyRosetta.'
                )
            )

    # BUG: every run returns the exact same results.
    def cart_ddg(self, p: DDGPayload) -> 'Mutant':

        scores = []

        # clone() is a shadow copy not a real deep copy.
        # deep_copy() is also clone()
        # https://graylab.jhu.edu/PyRosetta.documentation/pyrosetta.rosetta.core.pose.html#pyrosetta.rosetta.core.pose.Pose.detached_copy
        pose = pose_from_pdb(self.pose_path)
        newpose = deep_copy(pose)  # a detached copy

        mutant = p.mutant

        for i in range(p.iterations):
            scorefxn = create_score_function("ref2015_cart")
            newpose = mutate_repack_func4(
                pose=newpose,
                mutant=p.mutant,
                repack_radius=self.repack_radius,
                sfxn=scorefxn,
                verbose=False,
                cartesian=True,
                save_pose_to=p.save_pose_to_pdb(i),
                max_iter=self.relax_max_iter,
            )
            news = scorefxn(newpose)
            print(f'{mutant.id}.{i}: {news}')
            scores.append(news)

        mutant.scores = scores
        print(f'{str(mutant)}')
        return mutant

    def run(self, mutants=str) -> list[Mutant]:
        inputs_mutants = tuple([m for m in mutant_parser(mutants_str=mutants)])
        inputs_ = setup_ddg_payload(
            mutants=inputs_mutants,
            repeat_times=self.repeat_times,
            save_to=self.save_to,
        )

        with timing('Cartesian ddG'):
            with multiprocessing.Pool(
                processes=min(self.nproc, len(inputs_))  # , initializer=reseed
            ) as pool:
                results = pool.starmap(self.cart_ddg, inputs_)

        return results

    @staticmethod
    def summary(lm: list[Mutant]) -> pd.DataFrame:
        return pd.concat([m.asdataframe for m in lm], axis=0)
        