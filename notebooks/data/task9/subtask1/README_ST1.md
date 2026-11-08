# SemEval 2027 Task 9 — Subtask 1: Framing in Text

## First Training Data Release

This package contains the first batch of data released for **Subtask 1 (ST1): Detection and Classification of Framings in Text** of **SemEval 2027 Task 9 — Analysis of Multimodal Framing in the News**.

SemEval 2027 Task 9 studies how news articles frame events and issues across textual and visual modalities. In ST1, systems are asked to identify the framing(s) expressed by each paragraph of a news article. A paragraph may express more than one framing, making ST1 a **paragraph-level multi-label classification task**.

This initial release is intended to let participants become familiar with the data format, annotation scheme, character-offset conventions, and multilingual nature of the task, and to begin developing and testing their processing and modelling pipelines. **It should not be interpreted as the complete ST1 training dataset.** Additional training material will be released subsequently.

## Languages

The task covers four languages:

- **BG** — Bulgarian
- **EN** — English
- **PT** — European Portuguese
- **RU** — Russian

**Note:** The SemEval-2023 data covers nine languages, only two of which overlap with those considered here. Nevertheless, participants are free to use this data as they see fit.


## Limitations

### Representation of Langauges
In this first batch of data, some languages are not represented and other languages are present that are not part of the competition. 

### Important Note About `NONE` Instances

**This first batch does not contain `NONE` instances for Subtask 1.**

In other words, every paragraph included in the current ST1 annotations has at least one framing label assigned to it. Therefore, the distribution in this initial package is **not representative of the final task distribution with respect to paragraphs that do not express any of the predefined framings**.

Participants should take this into account when designing or evaluating systems using this release. In particular, the absence of `NONE` examples in this batch should **not** be interpreted as meaning that every paragraph in the full task data will necessarily receive a framing label. Later data releases may contain such negative instances.

## Framing Taxonomy

ST1 uses the following 14 generic framing categories:

1. `Economic`
2. `Capacity_and_resources`
3. `Morality`
4. `Fairness_and_equality`
5. `Legality_Constitutionality_and_jurisprudence`
6. `Policy_prescription_and_evaluation`
7. `Crime_and_punishment`
8. `Security_and_defense`
9. `Health_and_safety`
10. `Quality_of_life`
11. `Cultural_identity`
12. `Public_opinion`
13. `Political`
14. `External_regulation_and_reputation`


## Scope of This Release

This is an **early data release intended primarily for familiarisation and initial experimentation**. It enables participants to:

- implement readers and preprocessing pipelines before the complete training data are available;
- explore the 14-label framing taxonomy;
- test paragraph-level multi-label classification approaches;

Participants should avoid drawing conclusions about the final class distribution, language balance, frequency of `NONE` instances, or expected evaluation-set composition from this initial batch alone.

## Task Information

For the complete task description, updates, and subsequent data releases, please refer to the official SemEval 2027 Task 9 — **Analysis of Multimodal Framing in the News** task page.
