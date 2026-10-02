"""Offline validation harness for silent_killer_review.py.

`kaggle_benchmarks` is not installed in this environment (and the Model Proxy
needs live Kaggle creds), so this module stubs the SDK surface the task file
actually touches and then EXECUTES the real task file end-to-end. It verifies:

  * the DataFrame shape/columns and that they match the task's parameter names
  * the exact prompt text sent to the tested LLM
  * that the answer (`expected_bug`) is NEVER leaked into that prompt
  * that the Judge LLM receives a DIFFERENT rubric per row (core requirement)
  * that `.evaluate()` is wired to the DataFrame with n_jobs=3
  * that a model which misses the bug scores False WITHOUT erroring
"""
import contextlib
import inspect
import io
import pathlib
import sys
import types

import pandas as pd

TASK_FILE = pathlib.Path(
    r"c:\Users\BALAJI\OneDrive\Documents\kaggle_benchmark\silent_killer_review.py"
)

CAPTURED = {"prompts": [], "rubrics": [], "evaluate": None, "judge_calls": 0}
JUDGE_MODE = ["pass"]
STACK: list = []  # one assertion list per in-flight run


class AssertionResult:
    def __init__(self, passed, expectation=None):
        self.passed = bool(passed)
        self.expectation = expectation


class FakeLLM:
    def __init__(self, name, reply):
        self.name = name
        self.reply = reply

    def prompt(self, text, **kwargs):
        CAPTURED["prompts"].append(text)
        return self.reply


class FakeRun:
    def __init__(self, params, result, assertions, param_id=None):
        self.params = params
        self.result = result
        self.assertion_results = assertions
        self.param_id = param_id
        self.id = f"Run #{param_id}"

    @property
    def passed(self):
        return bool(self.result) and all(a.passed for a in self.assertion_results)


class FakeRuns:
    def __init__(self, runs):
        self.runs = runs

    def __iter__(self):
        return iter(self.runs)

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, i):
        return self.runs[i]


class FakeTask:
    def __init__(self, func, name=None, description=None, **kwargs):
        self.func = func
        self.name = name or func.__name__
        self.description = description

    def run(self, llm, **kwargs):
        bound = inspect.signature(self.func).bind(llm, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
        STACK.append([])
        try:
            result = self.func(**params)
        finally:
            assertions = STACK.pop()
        return FakeRun(params, result, assertions)

    def evaluate(self, llm=None, evaluation_data=None, n_jobs=1, **kwargs):
        CAPTURED["evaluate"] = {
            "llm": llm,
            "evaluation_data": evaluation_data,
            "n_jobs": n_jobs,
            "extra": kwargs,
        }
        runs = []
        for i, (_, row) in enumerate(evaluation_data.iterrows()):
            run = self.run(llm[0], **{c: row[c] for c in evaluation_data.columns})
            run.param_id = i
            run.id = f"Run #{i + 1}"
            runs.append(run)
        return FakeRuns(runs)


# --------------------------------------------------------------------------- #
# Stub the kaggle_benchmarks SDK
# --------------------------------------------------------------------------- #
def _judge(criteria, response_text, judge_llm, **kwargs):
    CAPTURED["judge_calls"] += 1
    CAPTURED["rubrics"].append(list(criteria))
    mode = JUDGE_MODE[0]

    class _AR:
        def __init__(self, criterion):
            self.criterion = criterion
            self.passed = mode == "pass"
            self.reason = "stubbed judge verdict"

    class _Report:
        def __init__(self, results):
            self.results = results

    if mode == "none":  # judge call blew up -> SDK returns None
        return None
    if mode == "empty":  # judge replied but produced no usable criteria
        return _Report([])
    return _Report([_AR(c) for c in criteria])


def _assert_true(expr, expectation=None):
    res = AssertionResult(expr, expectation)
    STACK[-1].append(res)
    return res


def _assert_fail(expectation=None):
    res = AssertionResult(False, expectation)
    STACK[-1].append(res)
    return res


assertions_mod = types.ModuleType("kaggle_benchmarks.assertions")
assertions_mod.assess_response_with_judge = _judge
assertions_mod.assert_true = _assert_true
assertions_mod.assert_fail = _assert_fail

MODEL_UNDER_TEST = FakeLLM(
    "model-under-test",
    "The fatal flaw is data leakage: the StandardScaler is fit before the "
    "train/test split, so test-set statistics contaminate training.",
)
JUDGE = FakeLLM("judge-model", "irrelevant")

kbench_stub = types.ModuleType("kaggle_benchmarks")
kbench_stub.llm = MODEL_UNDER_TEST
kbench_stub.judge_llm = JUDGE
kbench_stub.assertions = assertions_mod


def _task(*dargs, **dkwargs):
    if dargs and callable(dargs[0]) and not dkwargs:
        return FakeTask(dargs[0])

    def _decorate(func):
        return FakeTask(func, **dkwargs)

    return _decorate


kbench_stub.task = _task
kbench_stub.benchmark = _task
sys.modules["kaggle_benchmarks"] = kbench_stub
sys.modules["kaggle_benchmarks.assertions"] = assertions_mod


# --------------------------------------------------------------------------- #
# Execute the real task file with the IPython-only `%choose` line stripped
# --------------------------------------------------------------------------- #
source = TASK_FILE.read_text(encoding="utf-8")
executable = "\n".join(
    line for line in source.splitlines() if not line.lstrip().startswith("%choose")
)
namespace = {"__name__": "__main__", "__file__": str(TASK_FILE)}

stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(executable, str(TASK_FILE), "exec"), namespace)

print("----- captured stdout from the task file -----")
print(stdout.getvalue())

# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
df = namespace["buggy_scenarios"]
assert isinstance(df, pd.DataFrame), type(df)
assert df.shape == (3, 2), f"expected 3x2, got {df.shape}"
assert list(df.columns) == ["code_snippet", "expected_bug"], list(df.columns)

# ---- snippet integrity ------------------------------------------------------ #
# The paste-safety pass that stripped `# ====` decoration lines must NOT have
# touched a single byte of the code snippets (they live inside triple-quoted
# strings, so a sloppy regex could have eaten real content).
assert df.loc[0, "code_snippet"].startswith("# heart_disease_baseline.py")
assert "X_scaled = scaler.fit_transform(X)          # <-- fit() sees the WHOLE dataset" in df.loc[0, "code_snippet"]
assert "print(\"Test accuracy:\", model.score(X_test, y_test))" in df.loc[0, "code_snippet"]
assert "95% of patients are healthy" in df.loc[1, "code_snippet"]
assert "score = accuracy_score(y_test, preds)" in df.loc[1, "code_snippet"]
assert '"number_of_cardiology_visits",   # <-- "enriched" feature' in df.loc[2, "code_snippet"]
assert "FEATURES[model.feature_importances_.argmax()]" in df.loc[2, "code_snippet"]
# no snippet may contain a decoration-only comment line any more
for snip in df["code_snippet"]:
    for line in snip.splitlines():
        body = line.strip()
        if body.startswith("#"):
            content = body[1:].strip()
            assert not (content and set(content) <= set("=-")), line

task_obj = namespace["silent_killer_review"]
params = list(inspect.signature(task_obj.func).parameters)
assert params == ["llm", "code_snippet", "expected_bug"], params
assert task_obj.name == "silent_killer_review", task_obj.name
assert inspect.signature(task_obj.func).return_annotation is bool

ev = CAPTURED["evaluate"]
assert ev is not None, ".evaluate() was never called"
assert ev["evaluation_data"] is df, "evaluate() did not receive buggy_scenarios"
assert ev["llm"] == [MODEL_UNDER_TEST], ev["llm"]
assert ev["n_jobs"] == 3, ev["n_jobs"]
assert ev["extra"] == {}, ev["extra"]

# ---- prompt fidelity + no answer leakage ---------------------------------- #
TEMPLATE = (
    "Review this ML pipeline for predicting heart disease. Identify the fatal "
    "methodological flaw and explain why it will fail in production. \n\nCode: {code_snippet}"
)
assert len(CAPTURED["prompts"]) == 3, len(CAPTURED["prompts"])
for (_, row), prompt in zip(df.iterrows(), CAPTURED["prompts"]):
    assert prompt == TEMPLATE.format(code_snippet=row["code_snippet"]), prompt[:160]
    assert row["expected_bug"][:40] not in prompt, "ANSWER LEAKED INTO PROMPT"
    assert prompt.startswith(TEMPLATE.split("\n\n")[0]), prompt[:80]
    body, code = prompt.split("\n\nCode: ", 1)
    assert "leakage" not in body.lower(), "rubric leaked into the prompt"
    assert "accuracy" not in body.lower(), "rubric leaked into the prompt"

# ---- the dynamic rubric: 3 rows -> 3 different judge criteria ------------- #
assert CAPTURED["judge_calls"] == 3, CAPTURED["judge_calls"]
rubrics = CAPTURED["rubrics"]
assert len(rubrics) == 3, len(rubrics)
assert all(len(r) == 3 for r in rubrics), [len(r) for r in rubrics]
assert rubrics[0] != rubrics[1] and rubrics[1] != rubrics[2], "rubrics not row-specific"

keys = [namespace["_resolve_bug_key"](b) for b in df["expected_bug"]]
assert keys == ["data_leakage", "wrong_metric", "target_leakage"], keys

J = [" ".join(r).lower() for r in rubrics]
# row 1 -> leakage vocabulary; must NOT offer the other rows' fixes
assert "fit before split" in J[0]
assert "f1 / roc-auc" not in J[0]
# row 1's distractor guard names the OTHER two bugs
assert "target leakage" in J[0] and "class imbalance" in J[0]
# row 2 -> metric vocabulary; distractor-guards against leakage
assert "f1 / roc-auc" in J[1]
assert "data leakage" in J[1]
assert "fit before split" not in J[1]
# row 3 -> proxy-feature vocabulary; distractor-guards against both other bugs
assert "proxy" in J[2]
assert "data leakage" in J[2]
assert "f1 / roc-auc" not in J[2]
# every rubric embeds its own row's expected_bug verbatim
for (_, row), crit in zip(df.iterrows(), rubrics):
    assert row["expected_bug"] in " ".join(crit)
# the distractor guard is present in all three
assert all("must not misdiagnose" in j for j in J)

# ---- unknown expected_bug degrades gracefully instead of crashing --------- #
unknown_key, unknown_criteria = namespace["_build_criteria"]("SOMETHING UNLABELLED")
assert unknown_key == "unknown", unknown_key
assert len(unknown_criteria) == 2, "an unlabelled row drops only the distractor guard"
assert "SOMETHING UNLABELLED" in unknown_criteria[0]

# ---- a model that MISSES the bug scores False without raising -------------- #
JUDGE_MODE[0] = "fail"
missed = task_obj.run(
    MODEL_UNDER_TEST,
    code_snippet=df.loc[0, "code_snippet"],
    expected_bug=df.loc[0, "expected_bug"],
)
assert missed.result is False, missed.result
assert missed.passed is False, missed.passed
assert len(missed.assertion_results) == 3, len(missed.assertion_results)
assert not any(a.passed for a in missed.assertion_results)
assert all("data_leakage" in a.expectation for a in missed.assertion_results)
JUDGE_MODE[0] = "pass"
caught = task_obj.run(
    MODEL_UNDER_TEST,
    code_snippet=df.loc[2, "code_snippet"],
    expected_bug=df.loc[2, "expected_bug"],
)
assert caught.result is True and caught.passed is True
assert all("target_leakage" in a.expectation for a in caught.assertion_results)

# ---- a BROKEN judge must never award a free pass --------------------------- #
# Regression test for the bug found on review: before the `not assessment.results`
# guard, an empty judge report skipped the scoring loop entirely and the task
# returned True -- a silent free pass.
for mode in ("none", "empty"):
    JUDGE_MODE[0] = mode
    broken = task_obj.run(
        MODEL_UNDER_TEST,
        code_snippet=df.loc[1, "code_snippet"],
        expected_bug=df.loc[1, "expected_bug"],
    )
    assert broken.result is False, f"judge mode {mode!r} gave a FREE PASS"
    assert broken.passed is False, f"judge mode {mode!r} marked the run as passed"
    assert len(broken.assertion_results) == 1, len(broken.assertion_results)
    assert not broken.assertion_results[0].passed
    assert "wrong_metric" in broken.assertion_results[0].expectation
JUDGE_MODE[0] = "pass"

# ---- resolver robustness --------------------------------------------------- #
assert namespace["BUG_KEY_PRIORITY"] == (
    "wrong_metric",
    "target_leakage",
    "data_leakage",
), namespace["BUG_KEY_PRIORITY"]
assert namespace["_resolve_bug_key"](None) == "unknown"
assert namespace["_resolve_bug_key"]("") == "unknown"
assert params[0] == "llm", "llm must remain the FIRST task parameter"

print("OK: all harness checks passed (dynamic rubrics, scoring path, broken-judge guard)")

