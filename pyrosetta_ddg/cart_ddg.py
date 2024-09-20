import contextlib
from dataclasses import dataclass, asdict, field
from abc import ABC
import multiprocessing
import random
import time
import copy
from typing import Callable, Literal, Optional, Union


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

import dask
import pyrosetta.distributed.dask
import pyrosetta.distributed.io as io
import pyrosetta.distributed.packed_pose as packed_pose
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
import pyrosetta.distributed.tasks.score as score

from dask_jobqueue import SLURMCluster
from dask.distributed import (
    Client,
    LocalCluster,
)

DDG_INIT_FLAGS = "-default_max_cycles 200 -missing_density_to_jump -ex1 -ex2aro -ignore_zero_occupancy false -fa_max_dis 9 -mute all"

init(DDG_INIT_FLAGS)


# a full deep copy of the src pose obj.
def deep_copy(pose: Pose) -> Pose:
    p = Pose()
    p.assign(pose)
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

    @property
    def id(self):
        return f"{self.pos}{self.aa}"


@dataclass
class Mutant:
    mutations: list[Mutation]
    saved_score: dict[int, float] = field(default_factory=dict)

    @property
    def scores(self) -> tuple[float]:
        return tuple(self.saved_score.values())

    @scores.setter
    def scores(self, scores: tuple[float]):
        self.saved_score = {i: v for i, v in enumerate(scores)}

    @property
    def id(self) -> str:
        return (
            "_".join(m.id for m in self.mutations) if self.mutations else 'WT'
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

        return copy.deepcopy(self)

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


class DDGPayloadBase: ...


@dataclass(frozen=True)
class DDGPayloadOne(DDGPayloadBase):
    mutant: Mutant
    idx: int
    save_pose_to_pdb: str


@dataclass
class DDGPayloadBatch(DDGPayloadBase):
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
    mutants: list[Mutant],
    repeat_times: int = 3,
    save_to: Optional[str] = None,
    use_batch: bool = False,
) -> list[Union[DDGPayloadOne, DDGPayloadBatch]]:
    if use_batch:
        payload = [
            DDGPayloadBatch(
                m,
                [
                    (
                        os.path.join(save_to, f'{m.id}.i.{i}.pdb')
                        if save_to is not None
                        else None
                    )
                    for i in range(repeat_times)
                ],
            )
            for m in mutants
        ]
    else:
        payload = [
            DDGPayloadOne(
                m,
                i,
                (
                    os.path.join(save_to, f'{m.id}.i.{i}.pdb')
                    if save_to is not None
                    else None
                ),
            )
            for i in range(repeat_times)
            for m in mutants
        ]
    print(f'Payload Number: {len(payload)}')
    print(f'{payload=}')
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
        return io.to_packed(io.pose_from_file(save_pose_to))

    # Cloning of the pose including all settings
    working_pose = io.to_pose(pose)

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
    if verbose:
        mm = movemap.create_movemap_from_pose(working_pose)
        print(mm)

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
        working_pose = io.to_packed(working_pose)
    return working_pose


@contextlib.contextmanager
def create_local_cluster(
    cluster_type: Literal['slurm', 'local'] = 'local', **kwargs
):

    def pop_kwarg(k: str, default=None):
        if k in kwargs:
            return kwargs.pop(k)
        else:
            return default

    if cluster_type == 'slurm':
        cluster = SLURMCluster(
            n_workers=pop_kwarg('n_workers', 1),
            cores=pop_kwarg('cores', 1),
            processes=pop_kwarg('processes', 1),
            job_cpu=pop_kwarg('job_cpu', 1),
            memory=pop_kwarg('memory', "3GB"),
            queue=pop_kwarg('queue', "short"),
            walltime=pop_kwarg('walltime', "02:59:00"),
            local_directory=pop_kwarg('local_directory', "./job"),
            extra=pyrosetta.distributed.dask.worker_extra(
                init_flags=DDG_INIT_FLAGS
            ),
        )
    elif cluster_type == 'local':
        cluster = LocalCluster(
            n_workers=pop_kwarg('n_workers', 1),
            threads_per_worker=1,
            # extra=pyrosetta.distributed.dask.worker_extra(init_flags=DDG_INIT_FLAGS)
        )
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


def process_with_cluster(
    func: Callable,
    inputs: list = None,
    nstruct: Optional[int] = None,
    client: Optional[Client] = None,
    **kwargs,
):
    print(f'{inputs=}')
    print(f'{nstruct=}')
    print(f'{kwargs=}')

    if isinstance(client, Client):
        if isinstance(nstruct, int):
            nstruct_tasks = [client.submit(func) for i in nstruct]
            return [task.result() for task in nstruct_tasks]
        if isinstance(inputs, (list, tuple, set)):
            nstruct_tasks = [client.submit(func, i) for i in inputs]
            return [task.result() for task in nstruct_tasks]

    if isinstance(nstruct, int):
        results = [dask.delayed(func)(i) for i in range(nstruct)]
    else:
        results = [dask.delayed(func)(i) for i in inputs]

    # https://docs.dask.org/en/latest/delayed-best-practices.html#compute-on-lots-of-computation-at-once
    return dask.compute(*results)


@dataclass
class ddGRelaxer:
    pdb_path: str
    save_to: str = 'relaxed'

    nstruct: int = 20
    nproc: int = os.cpu_count()

    relax_max_recycle: Optional[int] = 5
    relax_max_iter: Optional[int] = None

    lowest_score: Optional[float] = None
    lowest_delta_score: Optional[float] = None

    _init_score: float = None
    _pdb_prefix: str = None
    use_slurm: bool = False

    def __post_init__(self):
        os.makedirs(self.save_to, exist_ok=True)
        self._pdb_prefix = os.path.basename(self.pdb_path)[:-4]
        from pyrosetta.toolbox import cleanATOM

        cleaned_pdb_path = os.path.join(
            self.save_to, f'{self._pdb_prefix}.cleaned.pdb'
        )
        cleanATOM(self.pdb_path, cleaned_pdb_path)

        self.pdb_path = cleaned_pdb_path

        scorefxn = create_score_function("ref2015_cart")
        pose = pose_from_pdb(self.pdb_path)

        self._init_score = scorefxn(pose)
        if isinstance(self.lowest_delta_score, (float, int)):
            self.lowest_score = self._init_score + self.lowest_delta_score

    def satisfied_energy(self, score: float) -> bool:
        if not self.lowest_score:
            return False
        return score < self.lowest_score

    def _relax(self, struct_label: int) -> dict[str, Union[str, int, float]]:
        pose = io.pose_from_file(self.pdb_path)
        pose_relax = io.to_pose(pose)
        scorefxn = create_score_function("ref2015_cart")

        for i in range(self.relax_max_recycle):
            relaxed_path = os.path.join(
                self.save_to,
                f'{self._pdb_prefix}.relax.{struct_label:>03}.r{i:>03}.pdb',
            )
            if os.path.isfile(relaxed_path):
                pose_relax = io.to_packed(io.pose_from_file(relaxed_path))
                print(
                    f'[{struct_label}] Recover packed decoy from: {relaxed_path}'
                )

            else:

                relax = pyrosetta.rosetta.protocols.relax.FastRelax()
                relax.set_scorefxn(scorefxn)

                relax.cartesian(True)
                if self.relax_max_iter:
                    relax.max_iter(self.relax_max_iter)
                relax.apply(pose_relax)

                pose_relax.dump_pdb(relaxed_path)

            # save and return the pdb file
            if (
                self.satisfied_energy(
                    score := scorefxn(io.to_pose(pose_relax))
                )
                or i == self.relax_max_recycle - 1
            ):
                pose_relax = io.to_packed(pose_relax)
                return {
                    'decoy': relaxed_path,
                    'score': score,
                    'score_before ': self._init_score,
                    'struc_label': struct_label,
                }
            pose_relax = io.to_pose(pose_relax)

    def run(self):

        with timing('Cartesian ddG Relax'), create_local_cluster(
            cluster_type='slurm' if self.use_slurm else 'local',
            n_workers=min(self.nproc, self.nstruct),
        ) as client:

            results = process_with_cluster(
                self._relax, [i for i in range(self.nstruct)], client=client
            )
            # with multiprocessing.Pool(
            #     processes=min(self.nproc, self.nstruct)  # , initializer=reseed
            # ) as pool:
            #     results = pool.map(self._relax, [i for i in range(self.nstruct)])
            print(results)

            return results

    @staticmethod
    def summary(results: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(results)

    @staticmethod
    def pick_the_best(df: pd.DataFrame) -> str:
        return df.loc[df['score'].idxmin()]['decoy']


@dataclass
class ddGRunner:

    pdb_path: str

    save_to: str = 'save'
    repeat_times: int = 3
    nproc: int = os.cpu_count()
    repack_radius: int = 6

    verbose: bool = False
    relax_max_iter: Optional[int] = None

    use_slurm: bool = False
    use_batch: bool = True

    def cart_ddg_batch(self, p: DDGPayloadBatch):
        packed = io.pose_from_file(self.pdb_path)
        pose = io.to_pose(packed)

        scores = []
        # BUG: a new pose object must be cloned before called
        newpose = deep_copy(pose)

        mutant = p.mutant

        for i in range(p.iterations):
            scorefxn = create_score_function("ref2015_cart")
            newpose = mutate_repack_func4(
                newpose,
                p.mutant,
                6,
                scorefxn,
                verbose=False,
                cartesian=True,
                save_pose_to=p.save_pose_to_pdb(i),
            )
            news = scorefxn(io.to_pose(newpose))
            print(f'{mutant.id}.{i}: {news}')
            scores.append(news)

        mutant.scores = scores
        print(f'{str(mutant)}')
        return mutant

    # BUG: every run returns the exact same results.
    def cart_ddg_one(self, p: DDGPayloadOne) -> 'Mutant':

        # clone() is a shadow copy not a real deep copy.
        # deep_copy() is also clone()
        # https://graylab.jhu.edu/PyRosetta.documentation/pyrosetta.rosetta.core.pose.html#pyrosetta.rosetta.core.pose.Pose.detached_copy
        pose = io.pose_from_file(self.pdb_path)
        newpose = io.to_pose(deep_copy(io.to_pose(pose)))

        mutant = p.mutant.copy

        scorefxn = create_score_function("ref2015_cart")
        newpose = mutate_repack_func4(
            pose=newpose,
            mutant=p.mutant,
            repack_radius=self.repack_radius,
            sfxn=scorefxn,
            verbose=False,
            cartesian=True,
            save_pose_to=p.save_pose_to_pdb,
            max_iter=self.relax_max_iter,
        )
        news = scorefxn(io.to_pose(newpose))
        print(f'{mutant.id}.{p.idx}: {news}')

        mutant.scores = [news]
        return mutant

    def run(self, mutants=str) -> list[Mutant]:
        inputs_mutants = tuple([m for m in mutant_parser(mutants_str=mutants)])
        inputs_ = setup_ddg_payload(
            mutants=inputs_mutants,
            repeat_times=self.repeat_times,
            save_to=self.save_to,
            use_batch=self.use_batch,
        )

        with timing('Cartesian ddG'), create_local_cluster(
            cluster_type='slurm' if self.use_slurm else 'local',
            n_workers=min(self.nproc, len(inputs_)),
        ) as client:

            results: list[Mutant] = process_with_cluster(
                (
                    self.cart_ddg_one
                    if not self.use_batch
                    else self.cart_ddg_batch
                ),
                inputs=inputs_,
                client=client,
            )
            if not self.use_batch:
                results = [
                    m.squash([r for r in results if r.id == m.id])
                    for m in inputs_mutants
                ]

        return results

    @staticmethod
    def summary(lm: list[Mutant]) -> pd.DataFrame:
        return pd.concat([m.asdataframe for m in lm], axis=0)
