# Corrected project code

This is a revised copy of the supplied code for OpenNeuro **ds004511**, session 01, Gambling Game (GG), run 01. The original script names remain as small entry points. Shared modules now handle preparation, features, evaluation, neural models, and optional LLM requests.

The original source folder was left unchanged. The supplied CSVs are preserved in `legacy_tables/`, with checksums and an explicit **unverified** status. The old plots are omitted because their source calculations and provenance cannot be trusted.

## Install and check

Use Python 3.12 in a virtual environment:

```sh
python -m venv .venv
```

Activate it using your operating system's normal command, then install the dependencies. The neural requirements include the core dependencies and are needed for the full regression suite:

```sh
python -m pip install -r requirements-neural.txt
python run_checks.py
```

`requirements-tested.txt` records the exact versions used for the delivered offline checks. `requirements.txt` is sufficient for classical/spatial analysis. LLM and CatBoost/XGBoost dependencies have separate optional requirements files. No script installs packages or downloads model weights automatically.

## What was corrected

- Trials retain their original participant/trial identifiers through exclusions, epoching, and feature extraction.
- EEG and physiological windows use explicit, checked onsets. Invented 2.5-second intervals and first-Sync anchoring were removed.
- Actual sampling rates, physiological `Columns`/`StartTime`, and named EEG channels are used. EOG, metadata-marked bad channels, and channels unavailable across participants are excluded explicitly.
- Continuous EEG has buffered filtering and antialiased resampling. Grid rounding, excluded trials, source metadata, and onset phase are recorded.
- EEG features are extracted in batches to limit memory use. Anatomical regions are no longer guessed from column positions.
- Short physiological windows produce accurately named amplitude statistics. They no longer claim HRV RMSSD, breathing rate, or detected skin-conductance responses.
- Missing physiological samples remain missing and carry QC fields. Failed extraction cannot silently create a zero-filled participant.
- Behavioral predictors use an RT whitelist and require explicit opt-in during raw preparation. Task outcomes, balances, correctness fields, trial order, and onset timestamps are excluded from predictors.
- Imputation, scaling, constant-feature removal, optional PCA, CSP, and xDAWN are fitted inside training folds. Held-out participants' labels never construct their features.
- Classical hyperparameters and optional thresholds are selected using grouped validation inside each outer training fold. Outer evaluation is leave-one-participant-out.
- Late fusion uses fixed equal weights. Stacking uses consistently ordered, grouped out-of-fold base predictions with nested fitting.
- A claimed modality must contain observed, varying training features. Empty or flat branches cannot silently become apparent multimodal evidence.
- Features unsupported by a training fold cannot affect held-out predictions. Neural inputs are masked, LLM prompts omit those columns, and per-fold feature QC records the decision.
- Neural inputs have accurate modality names. Contrastive learning and distillation include otherwise comparable baselines; attention summaries include all held-out participants.
- All methods use the same score definitions and preserve individual predictions. Failed LLM requests remain missing and reduce coverage; they never become random labels.
- Embedded credentials were removed. LLM models/providers are explicit, credentials come from an environment variable, responses are parsed strictly, and token parameters are configurable.
- Imports do not start training or contact an API. Comments and documentation use concise English.

## Prepare the raw dataset

The supplied folder contains no raw recordings. Its legacy tables cannot recover the original signal timing or establish how supervised spatial filters were fitted. Rebuild from the original ds004511 BIDS files before reporting scientific results.

First create an alignment template:

```sh
python -m deception alignment-template --dataset-root /path/to/ds004511 --output alignment.csv
```

Fill the template using verified acquisition/task timing. Each row must contain:

| Field | Meaning |
|---|---|
| `Subject_ID` | Original participant identifier, including `sub-`. |
| `Trial_ID` | Original `Trial_Order`, preserved even when other trials are excluded. |
| `EEG_Onset` | Chosen event time in seconds from EEG recording sample zero. |
| `Physio_Onset` | Same event on the physiological BIDS reference clock; physiology row zero is at JSON `StartTime`. |
| `Event_Phase` | One consistent event definition, such as `answer_onset` or `trial_start`. |

There are several indistinguishably named Sync markers per trial. The code deliberately requires a checked mapping rather than assuming the first marker identifies the first labeled trial. A template has blank onsets; it is not an inferred alignment. Record how the mapping was obtained. If the mapping is unavailable, the scientifically valid next step is to resolve it from the task/acquisition protocol.

Then prepare an empty output directory:

```sh
python -m deception prepare --dataset-root /path/to/ds004511 --alignment alignment.csv --output prepared
```

Optional `--subjects sub-S200116 ...` selects an explicit cohort. The documented participant with no GG EEG recording is excluded with a recorded reason. Every other missing required recording/metadata file produces a clear error. `dropped_trials.csv` lists exclusions by original identifiers.

Preparation defaults to an offline window of -0.2 to 1.5 seconds, a 1–45 Hz EEG filter with continuous context, and 256 Hz output. These are declared analysis choices, not proof that this window or filter is optimal. Zero-phase filtering uses future context, so this is **offline analysis**, not an online detection demonstration. Basic QC does not establish that remaining recordings are free from ocular or muscle artifacts.

Behavioral reaction times are excluded by default. Use `--include-behavior` only for a separately described post-response analysis: all trial RTs are available after the responses, and should not be presented as early prediction. Trials with unusable physiological windows retain missing values and QC; review the QC report before choosing the final cohort. Training-fold imputation cannot repair a wrong onset or poor acquisition.

The prepared bundle contains `trials.csv`, `epochs.npy`, and `metadata.json`, plus exclusion information. Metadata stores the label definition, event phase, actual window/grid, channel exclusions, source metadata hashes, alignment hash, and large recording paths/sizes. The large raw signal files are explicitly recorded as unhashed. Bundle checksums detect changed artifacts; they do not verify the science of the supplied alignment.

## Evaluate the models

Classical baselines and available modality combinations:

```sh
python -m deception benchmark --input prepared --output results/classical
```

This evaluates logistic regression, histogram gradient boosting and random forests, plus a majority baseline. Hyperparameter grids are fixed in the code and selected by inner grouped cross-validation. `--models`, `--feature-sets`, and `--inner-splits` make the comparison explicit. Available feature sets are `eeg`, `physiology`, `behavior`, `eeg_physiology`, `eeg_behavior`, and `fusion`; a named combination requires its constituent modalities.

Optional `--pca-components 0.95` fits PCA separately in each training fold. A prespecified grid such as `--thresholds 0.4 0.5 0.6` selects a threshold using inner validation only. Do not repeatedly inspect outer results to tune the experiment. Use at least three participants for nested learned models; each training fold needs both classes. Some cohort/group arrangements require more participants.

Spatial features and signal fusion:

```sh
python -m deception spatial --input prepared --output results/csp --representation csp
python -m deception spatial --input prepared --output results/xdawn --representation xdawn
python -m deception fusion --input prepared --output results/late --method late --feature-sets eeg physiology
python -m deception fusion --input prepared --output results/stacking --method stacking --feature-sets eeg physiology
```

CSP uses regularized class covariances. xDAWN uses MNE's public implementation and a correctly bounded ERP baseline. Both require aligned epochs and a common channel order, not precomputed legacy CSP columns. Stacking requires enough groups to perform tuning inside the out-of-fold base fits (at least four participants with the default three inner splits; more may be necessary with other splits/class distributions).

Neural comparisons:

```sh
python -m deception neural --input prepared --output results/attention --method attention --secondary physiology
python -m deception neural --input prepared --output results/contrastive --method contrastive --secondary physiology
python -m deception neural --input prepared --output results/distillation --method distillation --secondary physiology
```

The neural training schedule is fixed before outer testing. `--secondary behavior` selects the separate post-response behavioral analysis. Attention weights describe model routing; they do not demonstrate physiological causation. A strong result needs comparisons on identical participant/trial cohorts and a suitable uncertainty analysis across people.

Optional modern tree candidates:

```sh
python -m pip install -r requirements-boosters.txt
python -m deception benchmark --input prepared --output results/boosters --models logreg hist_gb catboost xgboost
```

CatBoost and XGBoost use the same grouped evaluation machinery and fixed candidate grids. Their training was not executed in the delivery environment because these optional packages were absent. Missing-dependency handling was tested. Their availability does not establish a state-of-the-art result.

## Results and reproducibility

Every evaluation writes:

- `predictions.csv`: original keys, labels, probabilities where available, predictions, model and feature set.
- `metrics.json`: pooled accuracy, balanced accuracy, macro F1, AUC where supported, coverage and per-participant scores.
- `selection.csv` / `folds.csv` where applicable: training participants, selected settings and thresholds.
- `run.json`: configuration, package versions, input provenance and code/output hashes.
- Attention summaries for relevant neural runs.
- `feature_qc.csv` for neural/LLM runs, including training support for each feature; attention summaries also count the participants with supported measurements.

The writer checks that every model attempts the same trial identities and uses the input labels. Explicitly sampled LLM runs record their sampling fraction and diagnostic status. AUC remains unavailable for an LLM that returns labels without probabilities. Results directories are not overwritten.

## Inspect the supplied legacy tables

```sh
python -m deception audit --input legacy_tables/processed_features_v9.csv --output legacy_audit.json
```

The saved tables have 4,751 trials from 33 participants, but lack verified original trial identities and processing provenance. The spatial CSV also differs from the supplied exporter. Legacy CSP/xDAWN columns are therefore excluded from ordinary feature selection. The tables have no recorded ECG/EDA/EMG feature blocks, so they cannot support those physiological fusion claims.

An explicitly labeled diagnostic can test software operation on the v9 table:

```sh
python -m deception benchmark --input legacy_tables/processed_features_v9.csv --output results/legacy_diagnostic --allow-unverified --models logreg --feature-sets eeg behavior eeg_behavior
```

The delivered `validation/legacy_v9_diagnostic/` run uses this mode. Its numbers are **not** a validated dataset benchmark and cannot show that the original alignment was correct. The copied `features_matrix.csv` is also unverified; the original folder contains a mock-data generator. It is never selected automatically as real data.

## Optional LLM experiment

Install `requirements-llm.txt`, set `OPENAI_API_KEY` outside the source, and explicitly choose the model/provider. Running this command sends the selected numeric features to that service and may incur API charges:

```sh
python -m deception llm --input prepared --output results/llm --model YOUR_MODEL --mode few_shot
```

Use `--base-url` for a compatible provider. For models requiring the newer token argument, use `--token-limit-parameter max_completion_tokens`; `--max-output-tokens` controls the budget. Temperature is omitted unless explicitly supplied. `--max-trials` opts into a clearly labeled diagnostic sample; the default attempts the full cohort. Few-shot examples and numeric preprocessing use other participants only. No live API calls were made during validation.

## Synthetic checks

```sh
python -m deception mock --output synthetic_control
python -m deception benchmark --input synthetic_control --output results/control --allow-synthetic --models logreg
```

These controls are labeled synthetic throughout. The regression suite tests training/test isolation, random-noise CSP, real MNE xDAWN execution, keyed middle-trial drops, physiological StartTime and chunk boundaries, anti-aliased EEG extraction, missing/flat modalities, model comparisons, API failures and strict parsing, result coverage, serialization, and overwrite protection.

## State-of-the-art status

The delivered checks establish software behavior in the tested environment. They cannot establish state-of-the-art performance without the raw recordings, verified timing/labels, a frozen protocol, and matched comparisons against relevant published methods. General tabular leaderboards use different datasets and often different splitting assumptions, so they do not rank methods for this participant-held-out experiment.

Primary references checked during the revision:

- [Original dataset and acquisition description](https://github.com/OpenNeuroDatasets/ds004511)
- [MNE xDAWN](https://mne.tools/stable/generated/mne.preprocessing.Xdawn.html)
- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [CatBoost classifier API](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)
- [XGBoost classifier API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
- [TabArena, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)
