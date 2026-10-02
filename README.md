# Silent Killer Review

A Kaggle Benchmark task that tests whether a language model can find a fatal methodological flaw in an ML pipeline review. It presents three realistic-looking heart-disease examples and scores each model response with a Judge LLM.

## Scenarios

| Scenario | Flaw being tested | Why it matters |
| --- | --- | --- |
| Data leakage | `StandardScaler` is fitted before the train/test split. | Test-set statistics influence preprocessing, making the reported score optimistic. |
| Wrong metric | Accuracy is used on a target with 95% healthy and 5% sick patients. | A model that predicts every patient as healthy still gets 95% accuracy. |
| Target leakage | `number_of_cardiology_visits` is used as a predictor even though it is a downstream consequence of diagnosis. | The feature may encode the answer and will not be available at prediction time. |

## How scoring works

The tested model receives the code snippet, but not the expected answer. The task builds a separate judge rubric from that scenario's `expected_bug` value. Each rubric checks whether the response:

1. Identifies the specific flaw.
2. Explains its production impact.
3. Avoids misdiagnosing a different scenario's flaw.

Each criterion is recorded as an assertion. A wrong model answer is a failed score, not an execution error; genuine model or judge exceptions can still fail a run. The run log includes the per-criterion verdicts.

## Repository files

- [`silent_killer_review.py`](silent_killer_review.py): benchmark task, three scenario snippets, dynamic judge rubrics, batch evaluation, and report output.
- [`_validate_silent_killer_review.py`](_validate_silent_killer_review.py): offline validation harness that stubs the Kaggle SDK and exercises the task's scoring behavior.

No external dataset is required; the example pipelines are embedded in the task file.

## Requirements

- Python 3.10 or newer
- Kaggle CLI with Benchmarks support
- `kaggle-benchmarks` and `pandas`
- Kaggle account and credentials for remote runs

Install the Python packages and Kaggle CLI in your environment, then initialize the Benchmarks configuration:

```bash
python -m pip install kaggle kaggle-benchmarks pandas
kaggle b init -y
```

Initialization configures Model Proxy credentials and local environment settings. Treat generated `.env` files and credentials as secrets; do not commit them to GitHub.

## Run locally

The offline harness does not call Kaggle or a live model. Run it with:

```bash
python _validate_silent_killer_review.py
```

It checks the scenario data, prompt isolation, row-specific rubrics, scoring behavior, and handling of an empty or unavailable judge result. The harness currently contains an absolute Windows path in `TASK_FILE`; update that constant to your checkout path if you run it elsewhere.

The benchmark source includes `%choose`, an IPython/Kaggle runner magic, so `python silent_killer_review.py` is not the local validation command.

## Push and run on Kaggle

The task slug must match the name in the `@kbench.task` decorator (`silent_killer_review`):

```bash
kaggle b t push silent_killer_review -f silent_killer_review.py --wait
kaggle b t run silent_killer_review
```

To select a model explicitly, list available models and pass the desired canonical model slug:

```bash
kaggle b t models
kaggle b t run silent_killer_review -m gemini-3.5-flash --wait
```

Inspect run status and logs, or download completed outputs:

```bash
kaggle b t status silent_killer_review
kaggle b t log silent_killer_review
kaggle b t download silent_killer_review -o ./results
```

See the [Kaggle Benchmarks CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md) for more commands and options.