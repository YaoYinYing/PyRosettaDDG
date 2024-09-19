from pyrosetta_ddg.cart_ddg import ddGRunner


def main():
    runner = ddGRunner(
        pose_path="test/1ubq.pdb",
        save_to='save_runner_test-1',
    )
    lm = runner.run(
        mutants='57A,54C,76A_32T',
    )


if __name__ == '__main__':
    main()
