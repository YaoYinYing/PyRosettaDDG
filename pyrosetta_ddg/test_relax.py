import os
from pyrosetta_ddg import ddGRelaxer


def main():
    relaxer=ddGRelaxer(
        pdb_path="test/1ubq.pdb",
        save_to="relaxed_client",
        nstruct=4,
        nproc=os.cpu_count(),
        relax_max_recycle=5,
        lowest_score=None,
        lowest_delta_score=None,
    )

    res=relaxer.run()

    df=ddGRelaxer.summary(res)
    print(df)
    final_picked=ddGRelaxer.pick_the_best(df)

    print(final_picked)


if __name__ == "__main__":
    main()