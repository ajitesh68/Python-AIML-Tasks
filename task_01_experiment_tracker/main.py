from tracker import ExperimentTracker

tracker = ExperimentTracker()

tracker.start_run("test_experiment_1")

tracker.log_params({"lr":0.01, "epochs":10})

tracker.log_metrics({"accuracy":0.88,"loss":0.23})

tracker.end_run(save=True)