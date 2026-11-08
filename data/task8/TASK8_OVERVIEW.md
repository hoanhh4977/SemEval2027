# SemEval 2027 Task 8 — StereoQueerEval

**Identification of Stereotypes and Hate Speech Targeting the LGBTQIA+ Community**

The first shared task on identifying stereotypes and hate speech targeting the LGBTQIA+ community in online discourse, at SemEval 2027. The data are YouTube comments. A comment must be interpreted together with the title and description of its video, which give the context needed to decide whether the comment contains a stereotype or hateful content.

## Data

Languages:

- English (EN)
- Italian (IT)
- Dutch (NL)
- Persian (FA): development and test sets only

For each language, comments were collected from selected news and media channels using LGBTQIA+-related keywords. The keywords were manually translated and culturally adapted for each language.

Each instance has three text fields:

| Field | Meaning |
|---|---|
| `yt_title` | title of the YouTube video |
| `yt_description` | description of the video |
| `yt_comment` | user-generated YouTube comment |

> **Important:** titles, descriptions and comments may contain newlines (`\n`) inside quoted fields. Read the files with a proper TSV parser, for example `pandas.read_csv(path, sep="\t")` with default quoting, not line-by-line splitting.

File columns: `StereoQueerEval_id, yt_title, yt_description, yt_comment, stereotype, hate_speech, target`

Local layout: `data/task8/train/` (EN, IT, NL), `data/task8/dev/` (FA), `data/task8/test/` (empty for now).

---

## Subtask A: Stereotype Detection

Binary classification: does a YouTube comment contain a stereotype targeting the LGBTQIA+ community?

Labels:

- `yes`: the comment contains a stereotype targeting LGBTQIA+ people.
- `no`: the comment does not contain such a stereotype.

A stereotype is a generalized, oversimplified, or fixed belief about a group of people. Here the focus is on stereotypes targeting people in the LGBTQIA+ community.

**Metric:** macro-averaged F1 over the two classes (yes / no).

```
Macro-F1 = (F1_yes + F1_no) / 2
```

---

## Subtask B: Hate Speech Classification

Three-class classification of the comment:

- `yes_explicit`: explicit hateful content, including direct slurs, insults, or other overtly hateful expressions targeting LGBTQIA+ individuals or groups.
- `yes_implicit`: hateful content expressed indirectly, implicitly, or through coded language, including implied insults, derogatory stereotypes, and other forms of indirect hateful expression.
- `no`: neutral, supportive, or otherwise non-hateful content.

**Metric:** macro-averaged F1 over the three classes. Each class counts equally regardless of its frequency in the test set.

```
Macro-F1 = (F1_no + F1_yes_implicit + F1_yes_explicit) / 3
```

---

## Subtask C: Target and Identity Identification

Evaluated **only on comments annotated as hateful in the gold data (Subtask B)**. For each hateful comment, predict two components:

1. the **target type**: `individual` or `group`;
2. one or more **targeted LGBTQIA+ identities**.

**Annotation format:** the target type, an underscore, then the identity label(s). Multiple identity labels are separated by commas, and their order does not affect evaluation. A comment may target more than one identity.

- Hateful comment targeting gay and bisexual people: `group_g,b`
- Hateful comment targeting an individual lesbian person: `individual_l`

> The task description renders the lesbian label like a digit `1` (for example `individual_1`), but in the training data it is the letter `l`. Non-hateful rows have `target = none`. Counts in train (EN+IT+NL): `t` 1609, `lgbtqia+` 1562, `g` 784, `l` 274, `nb` 140, `b` 94, `i` 55, `q` 40, `a` 2, so `a` (Asexual/Aromantic) is extremely rare.

Identity labels (9):

| Label | Identity |
|---|---|
| `l` | Lesbian |
| `g` | Gay |
| `b` | Bisexual |
| `t` | Transgender |
| `q` | Queer/Questioning |
| `i` | Intersex |
| `a` | Asexual/Aromantic |
| `nb` | Non-binary |
| `lgbtqia+` | LGBTQIA+ community as a whole |

**Metric:** Target–Identity Macro-F1, combining two macro-averaged F1 scores at 50% each.

- **Target Macro-F1:** macro-F1 over the two target types (`group` / `individual`).
- **Identity Macro-F1:** macro-F1 over the nine identity labels, each treated as an independent binary classification label (multi-label).

```
Target-Identity Macro-F1 = 1/2 * Target Macro-F1 + 1/2 * Identity Macro-F1

Target Macro-F1   = (F1_group + F1_individual) / 2
Identity Macro-F1 = (1/9) * sum_{i=1..9} F1_identity_i
```

The two components contribute equally regardless of the frequency of target classes or identity labels in the test set.

---

## Notes

- All three subtasks use macro-F1, so class imbalance matters. Rare classes (for example `i`, `a`, `nb`) weigh as much as common ones.
- Persian (FA) appears only in dev and test, so the models need cross-lingual or zero-shot transfer from EN/IT/NL.
- Evaluation scripts are in `StereoQueerEval_SemEval-2027/Evaluation Scripts/` (`task_a_evaluation.py`, `task_b_evaluation.py`, `task_c_evaluation.py`). Baselines are described in `StereoQueerEval_SemEval-2027/BASELINES.md`.
