"""
Experiment Tracker - Task 01
============================

Tujhe yeh file likhni hai from scratch.

INSTRUCTIONS:
- Neeche sirf class/function SIGNATURES hain (naam aur parameters)
- Docstrings mein HINTS hain ki kya karna hai
- Implementation (actual code) TU likhega
- Agar atke toh README.md padh ya mujhse poochh

CONCEPTS COVERED:
- dataclasses
- OOP (class, __init__, methods)
- type hints (int, str, dict, list, Optional)
- json module (dump, load)
- logging module
- datetime module
- exception handling (try/except)

GOAL: Jab tu main.py run kare, sab kuch kaam kare bina error ke.
"""

# ============================================================
# STEP 1: Imports
# ============================================================
# Tujhe chahiye: json, logging, os, dataclasses, datetime
# Soch: kaunse specific items import karega? (e.g., from dataclasses import dataclass, asdict)
# Hint: typing se Optional bhi chahiye hoga

# --- YAHAN IMPORTS LIKH ---


# ============================================================
# STEP 2: Logger Setup
# ============================================================
# logging.basicConfig() call kar with:
#   - level=logging.INFO
#   - format mein time, level, aur message aaye
# Phir ek logger bana: logger = logging.getLogger(__name__)

# --- YAHAN LOGGER SETUP LIKH ---


# ============================================================
# STEP 3: Experiment Dataclass
# ============================================================
# @dataclass decorator laga ke ek class bana:
#
# class Experiment:
#     name: str
#     description: str
#     parameters: dict        ← model ke hyperparameters (lr, epochs, etc.)
#     metrics: dict            ← results (accuracy, loss, etc.)
#     created_at: str          ← timestamp string
#     status: str              ← "created", "running", "completed", "failed"
#
# Hint: metrics ko empty dict se initialize kar using field(default_factory=dict)
# Hint: status ko "created" se initialize kar
# Hint: created_at ko datetime.now().isoformat() se initialize kar

# --- YAHAN DATACLASS LIKH ---


# ============================================================
# STEP 4: ExperimentTracker Class
# ============================================================
# Yeh main class hai — experiments manage karegi

class ExperimentTracker:
    """
    ML Experiments ka tracker.
    
    Kaam:
    - Experiments create karna
    - Metrics log karna
    - JSON file mein save/load karna
    - Experiments list karna
    """

    def __init__(self, storage_path: str = "experiments.json") -> None:
        """
        Tracker initialize kar.
        
        TODO:
        - self.storage_path set kar (parameter se)
        - self.experiments: dict[str, Experiment] = {} (empty dict, naam se access ke liye)
        - Agar storage_path pe file already exist kare, toh load() call kar
        - Logger se INFO log kar: "Tracker initialized with storage: {path}"
        """
        pass  # ← YAHAN IMPLEMENT KAR

    def create_experiment(
        self, name: str, description: str, parameters: dict
    ) -> "Experiment":
        """
        Naya experiment create kar.
        
        TODO:
        - Check kar: agar name already self.experiments mein hai → raise ValueError
        - Naya Experiment object bana (dataclass use kar)
        - self.experiments[name] mein store kar
        - self.save() call kar taaki data persist ho
        - Logger se INFO log: "Created experiment: {name}"
        - Experiment object return kar
        
        Soch: kya hoga agar koi same naam se dubara experiment banaye?
        """
        pass  # ← YAHAN IMPLEMENT KAR

    def log_metric(
        self, experiment_name: str, metric_name: str, value: float
    ) -> None:
        """
        Existing experiment mein metric add kar.
        
        TODO:
        - Check kar: agar experiment_name exist nahi karta → raise KeyError
        - experiment.metrics[metric_name] = value set kar
        - self.save() call kar
        - Logger se INFO log: "Logged {metric_name}={value} for {experiment_name}"
        
        Soch: kya metric overwrite hona chahiye ya list mein append?
              Abhi simple rakho — overwrite karo. Baad mein improve karenge.
        """
        pass  # ← YAHAN IMPLEMENT KAR

    def get_experiment(self, name: str):
        """
        Ek experiment return kar by name.
        
        TODO:
        - Agar name exist nahi karta → raise KeyError with descriptive message
        - Experiment object return kar
        
        Return type kya hoga? → Experiment (type hint laga)
        """
        pass  # ← YAHAN IMPLEMENT KAR

    def list_experiments(self, status_filter=None):
        """
        Sab experiments ki list return kar.
        
        TODO:
        - Agar status_filter None hai → sab experiments return kar
        - Agar status_filter diya hai → sirf woh experiments return kar jinka status match kare
        - Return type: list of Experiment objects
        
        Hint: list comprehension use kar
        Type hints: status_filter ka type kya hoga? → Optional[str]
                    return type? → list[Experiment]
        """
        pass  # ← YAHAN IMPLEMENT KAR

    def save(self) -> None:
        """
        Sab experiments JSON file mein save kar.
        
        TODO:
        - Har experiment ko dict mein convert kar (dataclasses.asdict() use kar)
        - json.dump() se file mein likh (indent=4 for readability)
        - Logger se INFO log: "Saved {n} experiments to {path}"
        
        Hint: with open(self.storage_path, 'w') as f: use kar (context manager)
        Soch: pura dict kaise banega? → {name: asdict(exp) for name, exp in ...}
        """
        pass  # ← YAHAN IMPLEMENT KAR

    def load(self) -> None:
        """
        JSON file se experiments load kar.
        
        TODO:
        - Check kar: agar file exist nahi karti → return (kuch mat kar)
        - json.load() se data padh
        - Har dict ko wapas Experiment object mein convert kar → Experiment(**data)
        - self.experiments mein store kar
        - Logger se INFO log: "Loaded {n} experiments from {path}"
        
        Exception Handling:
        - json.JSONDecodeError → file corrupt hai, WARNING log kar
        - FileNotFoundError → ignore kar (file nahi hai, normal hai)
        
        Hint: try/except use kar
        """
        pass  # ← YAHAN IMPLEMENT KAR
