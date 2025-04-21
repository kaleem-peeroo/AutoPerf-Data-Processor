    def test_get_run_names(self):
        o_exp = Experiment(
            s_name="600SEC_1B_1PUB_1SUB_BE_MC_100LC",
            ls_csv_paths=[
                "run1/600SEC_1B_1PUB_1SUB_BE_MC_100LC.csv",
                "run2/600SEC_1B_1PUB_1SUB_BE_MC_100LC.csv",
                "run3/600SEC_1B_1PUB_1SUB_BE_MC_100LC.csv",
                "run4/600SEC_1B_1PUB_1SUB_BE_MC_100LC.csv",
            ],
        )

        ls_run_names = o_exp.get_run_names()

        pprint(ls_run_names)

        assert len(ls_run_names) == 4
        assert ls_run_names == ["run1", "run2", "run3", "run4"]
