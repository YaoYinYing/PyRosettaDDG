from pyrosetta_ddg.cart_ddg import ddGRunner


def main():
    runner = ddGRunner(
        pose_path="test/1ubq.pdb",
        mutants='57A,54C,76A_32T',
        save_to='save_runner_test',
    )
    lm = runner.run()


if __name__ == '__main__':
    main()
