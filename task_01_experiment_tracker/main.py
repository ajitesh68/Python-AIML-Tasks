"""
Main - Task 01 Test File
========================

Jab tu tracker.py implement kar le, toh yeh file run kar ke test kar.
Neeche ek example workflow hai — UNCOMMENT karke use kar.

Run: python main.py
"""

from tracker import ExperimentTracker

# ============================================================
# TEST 1: Basic Workflow
# ============================================================
# Uncomment karke test kar jab tracker.py ready ho:

# # 1. Tracker create kar
# tracker = ExperimentTracker(storage_path="my_experiments.json")
#
# # 2. Experiment create kar
# exp1 = tracker.create_experiment(
#     name="logistic_regression_v1",
#     description="Baseline logistic regression on iris dataset",
#     parameters={"C": 1.0, "max_iter": 100, "solver": "lbfgs"}
# )
# print(f"Created: {exp1.name} at {exp1.created_at}")
#
# # 3. Metrics log kar
# tracker.log_metric("logistic_regression_v1", "accuracy", 0.95)
# tracker.log_metric("logistic_regression_v1", "f1_score", 0.93)
#
# # 4. Ek aur experiment
# exp2 = tracker.create_experiment(
#     name="random_forest_v1",
#     description="Random forest with default params",
#     parameters={"n_estimators": 100, "max_depth": 5}
# )
# tracker.log_metric("random_forest_v1", "accuracy", 0.97)
#
# # 5. Experiments list kar
# all_exps = tracker.list_experiments()
# print(f"\nTotal experiments: {len(all_exps)}")
# for exp in all_exps:
#     print(f"  - {exp.name}: {exp.metrics}")
#
# # 6. Ek experiment fetch kar
# fetched = tracker.get_experiment("logistic_regression_v1")
# print(f"\nFetched: {fetched.name}, Metrics: {fetched.metrics}")

# ============================================================
# TEST 2: Persistence Check
# ============================================================
# Uncomment karke check kar ki data persist hota hai:

# # Naya tracker bana — same file se load hoga
# tracker2 = ExperimentTracker(storage_path="my_experiments.json")
# loaded_exps = tracker2.list_experiments()
# print(f"\nLoaded {len(loaded_exps)} experiments from file:")
# for exp in loaded_exps:
#     print(f"  - {exp.name}: status={exp.status}, metrics={exp.metrics}")

# ============================================================
# TEST 3: Error Handling
# ============================================================
# Uncomment karke check kar ki errors sahi handle hote hain:

# # Duplicate experiment
# try:
#     tracker.create_experiment("logistic_regression_v1", "duplicate", {})
# except ValueError as e:
#     print(f"\n✅ Caught expected error: {e}")
#
# # Non-existent experiment mein metric log karna
# try:
#     tracker.log_metric("nonexistent_model", "accuracy", 0.5)
# except KeyError as e:
#     print(f"✅ Caught expected error: {e}")
