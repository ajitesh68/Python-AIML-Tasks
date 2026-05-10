# Task 01: Experiment Tracker 🧪

## Kya Banana Hai?

Ek `ExperimentTracker` class jo ML experiments ka record rakhti hai:
- Experiment create karna (naam, description, parameters)
- Metrics log karna (accuracy, loss, etc.)
- Sab data JSON file mein save/load karna
- Experiments list karna with filters

## Kya Seekhoge Is Task Mein?

| Concept | Where Used |
|---------|------------|
| **OOP (Classes)** | `ExperimentTracker` class banana |
| **Type Hints** | Har function mein parameter aur return types |
| **Dataclasses** | `Experiment` data structure define karna |
| **File I/O** | JSON mein experiments save/load karna |
| **Logging** | Python `logging` module se structured logs |
| **datetime** | Timestamps manage karna |
| **Exception Handling** | Graceful error handling (try/except) |
| **Dictionary Operations** | Nested data manipulation |

## Files

| File | Purpose |
|------|---------|
| `tracker.py` | Main `ExperimentTracker` class — **TU LIKHEGA** |
| `main.py` | Usage examples — test karne ke liye |

## Step-by-Step Hints

### Step 1: Experiment Data Structure
- `dataclass` use kar ek `Experiment` define karne ke liye
- Fields: `name`, `description`, `parameters` (dict), `metrics` (dict), `created_at` (str), `status` (str)
- Soch: agar ye data JSON mein save karna hai, toh dataclass ko dict mein kaise convert karega?

### Step 2: ExperimentTracker Class
- `__init__` mein ek empty list/dict rakh experiments store karne ke liye
- `storage_path` parameter le taaki JSON file ka location set ho
- Logger setup kar `logging` module se

### Step 3: Methods Implement Kar
- `create_experiment(name, description, params)` → naya experiment banaye, list mein add kare
- `log_metric(experiment_name, metric_name, value)` → existing experiment mein metric add kare
- `get_experiment(name)` → ek experiment return kare
- `list_experiments(status_filter=None)` → sab experiments list kare, optional filter ke saath
- `save()` → sab data JSON file mein save kare
- `load()` → JSON file se data load kare

### Step 4: Error Handling
- Kya hoga agar experiment name already exist kare?
- Kya hoga agar non-existent experiment mein metric log kare?
- Kya hoga agar JSON file corrupt ho?

### Step 5: Test Kar `main.py` mein
- Tracker create kar, experiments add kar, metrics log kar
- Save kar, naya tracker create kar, load kar — verify data persist hua

## ✅ Done Kab Hoga?

- [ ] `ExperimentTracker` class likhna with type hints
- [ ] `Experiment` dataclass define karna
- [ ] JSON save/load working
- [ ] Logging lagana (INFO for actions, WARNING for errors)
- [ ] `main.py` mein test karke verify karna
- [ ] Git commit + push karna

## 💡 Pro Tips

- `json` module ka `indent=4` use kar readable output ke liye
- `datetime.now().isoformat()` se clean timestamp milega
- `logging.basicConfig()` se quick setup ho jayega, baad mein customize karenge
- `dataclasses.asdict()` bohot kaam aayega JSON conversion mein
