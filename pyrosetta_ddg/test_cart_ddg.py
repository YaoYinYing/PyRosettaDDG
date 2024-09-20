from pyrosetta_ddg import ddGRunner

def main():
    runner = ddGRunner(
        pdb_path="test/1ubq.pdb",
        save_to='save_runner_client_usebatch',
        use_batch=True
        
    )
    lm = runner.run(
        mutants='57A,54C,76A_32T',
    )

    s=runner.summary(lm)
    print(s)


if __name__ == '__main__':
    main()
