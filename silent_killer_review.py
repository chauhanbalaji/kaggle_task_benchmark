# %% [markdown]
# # Silent Killer Review
#
# A Kaggle Benchmark that tests whether an LLM can **review an ML pipeline and
# catch a fatal methodological flaw** -- the kind of bug that produces a green
# dashboard and a dead model in production.
#
# Three broken heart-disease pipelines are evaluated as three batch rows:
#
# | # | `expected_bug` family | The silent killer |
# |---|----------------------|-------------------|
# | 1 | DATA LEAKAGE | `StandardScaler` fitted on `X` **before** `train_test_split` |
# | 2 | WRONG METRIC | `accuracy_score` on a 95% healthy / 5% sick target |
# | 3 | TARGET LEAKAGE | `number_of_cardiology_visits` used as a **predictor** (it is a *result* of the label) |
#
# The scoring is done by a Judge LLM whose rubric is **built at runtime from the
# `expected_bug` argument of the current row** -- see `_build_criteria()`.

# %%
# 1. Imports
import textwrap

import pandas as pd

import kaggle_benchmarks as kbench

# %%
# 2. The three "silent killer" code snippets
# Each snippet is a realistic-looking pipeline that a model would happily
# approve of if it only read for syntax. They are kept as plain strings so the
# DataFrame is fully self-contained (no dataset downloads, no sklearn import).

# Row 1: DATA LEAKAGE -- scaler fitted on X BEFORE train_test_split
LEAKAGE_SCALER_BEFORE_SPLIT = textwrap.dedent(
    """
    # heart_disease_baseline.py -- "standardised" logistic-regression baseline
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv("heart_disease.csv")
    X = df.drop(columns=["has_heart_disease"])
    y = df["has_heart_disease"]

    # Standardise every feature so the solver converges faster.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)          # <-- fit() sees the WHOLE dataset

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("Test accuracy:", model.score(X_test, y_test))
    """
).strip()


# Row 2: WRONG METRIC -- accuracy on a 95% healthy / 5% sick target
WRONG_METRIC_ACCURACY_ON_IMBALANCE = textwrap.dedent(
    """
    # heart_disease_eval.py
    # Dataset note: this is a screening cohort -- 95% of patients are healthy
    # (has_heart_disease = 0) and only 5% actually have heart disease (= 1).
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    df = pd.read_csv("heart_disease_screening.csv")
    X = df.drop(columns=["has_heart_disease"])
    y = df["has_heart_disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = accuracy_score(y_test, preds)
    print("Model accuracy:", score)
    # 0.95 accuracy -- ship it!
    """
).strip()


# Row 3: TARGET LEAKAGE -- a post-diagnosis column used as a predictor
TARGET_LEAKAGE_PROXY_FEATURE = textwrap.dedent(
    """
    # heart_disease_with_visits.py -- RandomForest with the "enriched" feature set
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("heart_disease_enriched.csv")

    FEATURES = [
        "age",
        "sex",
        "resting_bp",
        "cholesterol",
        "max_heart_rate",
        "number_of_cardiology_visits",   # <-- "enriched" feature
    ]

    X = df[FEATURES]
    y = df["has_heart_disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    print("Validation accuracy:", model.score(X_test, y_test))
    print("Top feature:", FEATURES[model.feature_importances_.argmax()])
    """
).strip()


# %%
# 3. THE BENCHMARK DATASET -- 3 rows == 3 independent task evaluations
# Column names are load-bearing: `.evaluate()` binds DataFrame columns to the
# task's keyword parameters *by name*, so these two columns MUST stay in sync
# with `def silent_killer_review(llm, code_snippet, expected_bug)`.
buggy_scenarios = pd.DataFrame(
    [
        {
            "code_snippet": LEAKAGE_SCALER_BEFORE_SPLIT,
            "expected_bug": (
                "DATA LEAKAGE (leaky preprocessing): StandardScaler is fitted on the complete "
                "feature matrix X before train_test_split is called, so the training data "
                "absorbs the mean and standard deviation of the hold-out test set. The "
                "reported test score is optimistically biased and will collapse in production."
            ),
        },
        {
            "code_snippet": WRONG_METRIC_ACCURACY_ON_IMBALANCE,
            "expected_bug": (
                "WRONG METRIC: accuracy_score is the metric on a dataset that is 95% healthy "
                "(y=0) and 5% sick (y=1). A trivial majority-class model that always predicts "
                "'healthy' already scores 95% accuracy, so accuracy cannot reveal failures on "
                "the clinically critical minority class. F1 / ROC-AUC / recall (or "
                "precision-recall AUC) must be used instead."
            ),
        },
        {
            "code_snippet": TARGET_LEAKAGE_PROXY_FEATURE,
            "expected_bug": (
                "TARGET LEAKAGE (proxy feature): number_of_cardiology_visits is a downstream "
                "consequence of the disease -- it exists only because the patient was already "
                "sick -- so it encodes the target label. Using it as a predictor inflates the "
                "offline metric and the value is unavailable at real inference time, before a "
                "diagnosis has been made."
            ),
        },
    ]
)


# %%
# 4. DYNAMIC JUDGE RUBRIC -- how the Judge LLM's criteria change per row
# THE CENTRAL IDEA OF THIS BENCHMARK:
#
#   A single hard-coded rubric would leak the answer. If the judge prompt always
#   asked "did the model say data leakage AND wrong metric AND target leakage?",
#   then row 2 would be scored against a rubric that also demands the row-1 and
#   row-3 bugs -- every model would fail, and the benchmark would measure nothing.
#
#   Instead, the tested LLM never sees the rubric. Only the Judge LLM does, and
#   its criteria string is **assembled at task runtime from the `expected_bug`
#   argument of the row currently being evaluated**. Concretely, for each row:
#
#     * `_resolve_bug_key(expected_bug)` maps that row's prose onto one of the
#       three canonical bug families ("data_leakage" / "wrong_metric" /
#       "target_leakage").
#     * `_build_criteria(expected_bug)` then emits a 3-criterion rubric tailored
#       to that family:
#           1. IDENTIFY    -- the model must name THIS row's flaw, using
#                             family-specific accepted phrasings (row 1 accepts
#                             "fit before split"; row 3 accepts "proxy for the
#                             label"). Another family's vocabulary fails here.
#           2. IMPACT      -- the model must explain the family-specific
#                             production failure mode (biased test score /
#                             95%-baseline trap / feature absent at inference).
#           3. NO MISDIAGNOSIS -- a *distractor guard*. The roster of "wrong"
#                             answers for row 1 is literally rows 2 and 3's bugs,
#                             and vice versa. This is what separates "caught the
#                             silent killer" from "pattern-matched a plausible
#                             -sounding ML complaint".
#
#   Net effect per row: identical task code, three different rubrics.

# Canonical bug families. `markers` are matched case-insensitively against the
# row's `expected_bug` text; `accepted_phrasings` become the judge's pass-list;
# `distractors` become the judge's fail-list.
JUDGE_RUBRICS = {
    # Row 1: fit before split
    "data_leakage": {
        "markers": ("data leakage",),
        "accepted_phrasings": [
            "data leakage",
            "leakage",
            "the StandardScaler is fit / fitted before the train_test_split split",
            "'fit before split' / 'fit-then-split'",
            "preprocessing was fitted on the full dataset, including the test rows",
            "test-set statistics leaked into the training data",
            "the scaler was calibrated using the hold-out set",
            "training data contaminated by the evaluation data",
        ],
        "production_failure": (
            "the reported test accuracy is optimistically biased and will drop on genuinely "
            "unseen patients, because the scaler's mean/variance were estimated from the very "
            "rows later used to score the model"
        ),
        "distractors": [
            "class imbalance or the choice of accuracy as the metric",
            "target leakage from a proxy / post-diagnosis feature",
            "the model family or its hyper-parameters",
        ],
    },
    # Row 2: accuracy on a 95/5 target
    "wrong_metric": {
        "markers": ("wrong metric",),
        "accepted_phrasings": [
            "wrong metric",
            "inappropriate / misleading / uninformative metric",
            "accuracy is the wrong metric for imbalanced data",
            "class imbalance makes accuracy meaningless",
            "a majority-class baseline that always predicts 'healthy' already scores 95%",
            "should report F1 / ROC-AUC / recall / precision-recall AUC instead",
            "accuracy hides the false-negative rate on the sick patients",
        ],
        "production_failure": (
            "a model that predicts 'healthy' for every patient scores 95% accuracy while "
            "detecting zero sick patients, so the metric hides a catastrophic false-negative "
            "rate and a useless screening tool would be shipped as 'working'"
        ),
        "distractors": [
            "StandardScaler fitted before train_test_split / data leakage",
            "target leakage from a proxy / post-diagnosis feature",
            "the model family or its hyper-parameters",
        ],
    },
    # Row 3: proxy / post-diagnosis column
    "target_leakage": {
        "markers": ("target leakage",),
        "accepted_phrasings": [
            "target leakage",
            "label leakage",
            "the feature is a proxy / stand-in for the target",
            "'number_of_cardiology_visits' is a consequence or result of the disease, not a predictor",
            "the column is not available at prediction time / before diagnosis",
            "circular feature / circular reasoning -- the feature is caused by the outcome",
            "post-diagnosis or downstream variable",
            "the feature must be dropped from the training set",
        ],
        "production_failure": (
            "the offline validation score is meaningless because the feature encodes the "
            "answer, and the value does not exist yet for a new patient in production (it is "
            "only recorded after they are already diagnosed and treated), so live performance "
            "collapses"
        ),
        "distractors": [
            "StandardScaler fitted before train_test_split / data leakage",
            "accuracy being the wrong metric for class imbalance",
            "the model family or its hyper-parameters",
        ],
    },
}


# %%
# 5. Rubric assembly -- this is where the judge's criteria become row-specific

# Resolution priority: "wrong metric" and "target leakage" are tested BEFORE the
# generic "data leakage" marker, so the three families can never collide
# (row 3's "TARGET LEAKAGE" must not be read as row 1's "DATA LEAKAGE").
BUG_KEY_PRIORITY = ("wrong_metric", "target_leakage", "data_leakage")


def _resolve_bug_key(expected_bug: str) -> str:
    """Map a row's `expected_bug` prose onto a canonical bug family.

    This is the *switch* that makes the judge rubric dynamic: the same task code
    produces a different rubric for every row because `expected_bug` differs.

    Matching is case-insensitive and priority-ordered (see `BUG_KEY_PRIORITY`).
    Rubric entries are looked up with `.get()` so a marker listed in
    `BUG_KEY_PRIORITY` without a matching rubric degrades to the next family
    instead of raising a KeyError mid-evaluation.

    Input is coerced with `str(...)` so a `None`/NaN cell (a real possibility in
    a hand-edited DataFrame) resolves to "unknown" instead of raising
    AttributeError on `.lower()`.
    """
    normalized = str(expected_bug or "").lower()
    for bug_key in BUG_KEY_PRIORITY:
        rubric = JUDGE_RUBRICS.get(bug_key)
        if rubric and any(marker in normalized for marker in rubric["markers"]):
            return bug_key
    return "unknown"


def _build_criteria(expected_bug: str) -> tuple[str, list[str]]:
    """Build the (bug_key, criteria) pair the Judge LLM will score against.

    Returning the key as well lets the task tag its assertions with the scenario
    it was judging, which makes the Kaggle run log readable.

    The row's own `expected_bug` text is embedded VERBATIM in criterion 1, so
    even an unrecognised row is still graded on the right thing (graceful
    degradation) -- only the extra pass-list / fail-list hints are lost.
    """
    bug_key = _resolve_bug_key(expected_bug)
    rubric = JUDGE_RUBRICS.get(bug_key)

    # Criterion 1: did it name THIS row's flaw?
    # The accepted vocabulary is family-specific: row 1's rubric literally lists
    # "fit before split", row 2's lists "F1 / ROC-AUC", row 3's lists "proxy for
    # the target". So the judge cannot accept the wrong bug's vocabulary.
    accept_hint = (
        "Accept any of these equivalent formulations: " + "; ".join(rubric["accepted_phrasings"]) + ". "
        if rubric
        else ""
    )
    criteria = [
        "The response must explicitly identify the fatal methodological flaw described below "
        "and present it as THE primary reason the pipeline will fail. "
        f"Flaw as specified for this code: {expected_bug} "
        + accept_hint
        + "Strictness: a response that only says the code 'could be improved', 'may overfit', "
        "'should be validated more', or that describes a different flaw does NOT satisfy this "
        "criterion, even if the rest of the review is otherwise excellent."
    ]

    # Criterion 2: did it explain the production consequence?
    # Also family-specific: leakage => biased score, metric => 95% baseline trap,
    # proxy feature => value absent at inference time.
    impact = (
        rubric["production_failure"]
        if rubric
        else f"the concrete production consequence of the flaw stated above ({expected_bug})"
    )
    criteria.append(
        "The response must explain the concrete production failure mode: "
        f"{impact}. "
        "Strictness: merely restating the flaw without explaining why it breaks in "
        "production does NOT satisfy this criterion."
    )

    # Criterion 3: distractor guard -- the anti-"pattern match" check
    # For row 1 the distractors are rows 2 and 3's bugs, etc. This is what stops
    # a judge from passing a response that names a plausible-but-wrong flaw.
    if rubric:
        criteria.append(
            "The response must NOT misdiagnose the flaw. It fails this check if it presents "
            "any of the following as THE fatal flaw instead of the flaw in criterion 1: "
            + "; ".join(rubric["distractors"])
            + ". Strictness: briefly listing such issues as secondary or minor observations is "
            "acceptable, but only if criterion 1 was satisfied."
        )

    return bug_key, criteria


# %%
# 6. THE TASK -- one silent killer per invocation
# Signature contract: `(llm, code_snippet, expected_bug)`.
#   * `llm`            -- the model under test (always the FIRST parameter).
#   * `code_snippet`   -- must match the DataFrame column name of the same name.
#   * `expected_bug`   -- must match the DataFrame column name of the same name.
# `.evaluate()` binds columns -> parameters by name, so renaming either one
# silently breaks the batch run.
PROMPT_TEMPLATE = (
    "Review this ML pipeline for predicting heart disease. Identify the fatal "
    "methodological flaw and explain why it will fail in production. "
    "\n\nCode: {code_snippet}"
)


@kbench.task(
    name="silent_killer_review",
    description=(
        "Can the model catch the 'silent killer' bug in a heart-disease ML pipeline? "
        "Three scenarios: (1) StandardScaler fitted before train_test_split, (2) accuracy "
        "reported on a 95%/5% imbalanced target, (3) target leakage via a proxy feature. "
        "Scored by a Judge LLM whose criteria are built from the row's `expected_bug`."
    ),
)
def silent_killer_review(llm, code_snippet, expected_bug) -> bool:
    """Returns True only if the Judge LLM confirms every criterion for this row."""
    # Step 1: ask the model under test to review the pipeline
    # NOTE: only the code is interpolated here. `expected_bug` and the rubric are
    # NEVER sent to the tested LLM -- it has to find the bug on its own.
    prompt = PROMPT_TEMPLATE.format(code_snippet=code_snippet)
    response = llm.prompt(prompt)

    # Step 2: build THIS row's judge rubric
    # This is the dynamic part: row 1 -> leakage rubric, row 2 -> metric rubric,
    # row 3 -> proxy-feature rubric. Same task, different grading criteria.
    bug_key, criteria = _build_criteria(expected_bug)

    # Step 3: Judge LLM grades the free-text review
    # `assess_response_with_judge` runs the judge in its own chat, returns an
    # AssessReport (or None if the judge errored), and must be None-checked.
    assessment = kbench.assertions.assess_response_with_judge(
        criteria=criteria,
        response_text=response,
        judge_llm=kbench.judge_llm,
    )

    # An empty `results` list is treated exactly like a `None` report. Without
    # this guard the `for` loop below would never execute, `all_criteria_passed`
    # would stay True, and a malformed judge response would award a FREE PASS --
    # precisely the kind of silent green-dashboard bug this benchmark hunts.
    if assessment is None or not assessment.results:
        kbench.assertions.assert_fail(
            expectation=(
                f"Judge LLM returned no usable assessment for scenario '{bug_key}' "
                f"(assessment={assessment!r}), so the review could not be scored and "
                "must not count as a catch."
            )
        )
        return False

    # Step 4: convert each judged criterion into a recorded assertion
    # Every criterion fails independently, so the run log shows exactly which
    # part of the review was missing (identify / impact / no-misdiagnosis).
    all_criteria_passed = True
    for judged in assessment.results:
        kbench.assertions.assert_true(
            judged.passed,
            expectation=(
                f"[{bug_key}] {judged.criterion} -- judge reason: {judged.reason}"
            ),
        )
        all_criteria_passed = all_criteria_passed and judged.passed

    # `-> bool` return type => `results.as_dataframe().result.mean()` is the score.
    return all_criteria_passed


# %%
# 7. BATCH EVALUATION -- 3 rows -> 3 runs of the task
# `.evaluate()` expands `buggy_scenarios` into one task invocation per row and
# binds columns to parameters by name, i.e. it effectively calls:
#     silent_killer_review.run(llm, code_snippet=row["code_snippet"],
#                                   expected_bug=row["expected_bug"])
# for each of the 3 rows -- which is what mints the dynamic judge rubric.
#
# `n_jobs=3` runs the three reviews in parallel (threads), so wall-clock time is
# roughly that of a single review.
#
# NOTE: `llm=` takes a SEQUENCE of models, because the SDK scheduler expands it as
# a grid dimension (`itertools.product`), so each run receives ONE model object --
# which is why `llm.prompt(...)` works inside the task. `[kbench.llm]` therefore
# yields 1 model x 3 rows = 3 runs; append more models to multiply that.
#
# NOTE: deliberately NO `stop_condition`. In this SDK version it is only consulted
# *between* retry attempts (i.e. only when `max_attempts > 1` AND at least one run
# already failed), so with the default `max_attempts=1` it is dead code that would
# misleadingly suggest the batch size is bounded by it.
MODELS_UNDER_TEST = [kbench.llm]

results = silent_killer_review.evaluate(
    llm=MODELS_UNDER_TEST,
    evaluation_data=buggy_scenarios,
    n_jobs=3,
)

# %%
# Reporting
# NOTE on failure semantics: a model that MISSES a bug is not an error. Assertion
# failures are recorded (passed=False) and leave the run status as SUCCESS, so
# `on_failure="raise"` never aborts the batch for a wrong answer. Only genuine
# exceptions (e.g. a judge/model API error) would mark a run FAILED.
summary_rows = []
for run in results:
    row_expected_bug = run.params.get("expected_bug", "")
    summary_rows.append(
        {
            "scenario": str(row_expected_bug)[:52] + "...",
            "judge_rubric": _resolve_bug_key(row_expected_bug),
            "model_caught_it": bool(run.passed),
        }
    )

summary = pd.DataFrame(
    summary_rows, columns=["scenario", "judge_rubric", "model_caught_it"]
)

if summary.empty:
    # e.g. every run errored before producing a result. Fail gracefully instead of
    # raising a KeyError on the (missing) 'model_caught_it' column.
    print("No completed runs to report -- inspect the errored runs above.")
else:
    print(summary.to_string(index=False))
    print(
        f"\nSilent-killer detection rate: {summary['model_caught_it'].mean():.0%} "
        f"({int(summary['model_caught_it'].sum())}/{len(summary)})"
    )

# Per-criterion verdicts -- read these in the Kaggle run log to see exactly what
# the judge rejected (missing identification vs. missing impact vs. misdiagnosis).
for run in results:
    print(f"\n=== scenario {run.id} -> {'CAUGHT' if run.passed else 'MISSED'} ===")
    for judged in run.assertion_results:
        print(f"  [{'PASS' if judged.passed else 'FAIL'}] {judged.expectation[:200]}")

# %%
# 8. LEADERBOARD ENTRY POINT
# `%choose` tells the Kaggle Benchmarks runner which task in this notebook feeds
# the leaderboard. It is an IPython magic that resolves against the
# `@kbench.task(name=...)` defined above.
#
# IMPORTANT: `%choose` is only valid inside the IPython/Kaggle notebook runtime.
# A plain `python silent_killer_review.py` smoke test will raise a SyntaxError on
# this line. For a local script run, comment it out (or swap in the guarded form
# below, which is equivalent but importable):
#
#     try:
#         get_ipython().run_line_magic("choose", "silent_killer_review")
#     except NameError:          # not running under IPython
#         pass
#
%choose silent_killer_review


