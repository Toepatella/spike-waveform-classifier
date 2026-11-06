# Analysis plan

Rules fixed before the analyses they govern. Each rule changes only with a dated note saying why.
Objective 1 is the classification of the monkey neurons by spike shape; Objective 2 is the
comparison of iPSC-derived neurons with those classes.

## Waveform features

The definitions in the README's "Waveform features" section, as measured by `features.py`, will be
fixed, together with the feature set Objective 1 uses, before its clustering. Split-half
reliability in the 217 main units of PFC, CA3 and DG of the 2026-07-21 session (templates rebuilt
from odd and even stored spikes, Spearman correlation projected to full templates): 0.87–1.00 for
every feature except `repolarization_ms` (0.40).

Decided on 2026-11-06:

- The baseline (the median of the template's first 0.5 ms), the interpolation filter (Kaiser
  window, β 10) and the peak-time rule (the centre of the samples at or above 90% of the maximum)
  stay as defined in the README.
- The intermediate-class check's blends are measured on the feature channel (step 5 below).
- `repolarization_ms` stays in the feature table but is not in the clustering feature set: with a
  split-half reliability of 0.40, it mostly measures noise on 500-spike templates.

## Intermediate-class check

Fixed on 2026-11-05, before any clustering. A unit that has picked up another neuron's spikes
averages to a blended waveform, which can sit between narrow and broad. The check asks whether an
intermediate-width class could come from such units.

Terms:

- *Likely contaminated*: `rp_contamination_lower` above 0.20.
- *Testable*: a unit with enough spikes that it could be flagged as likely contaminated while
  still passing the 0.5 cut. With the largest whole number of refractory pairs that keeps
  `rp_contamination` below 0.5, `rp_contamination_lower` is above 0.20 (worked out with the
  formulas in `good_cells.py` and each probe's length and sampling rate). Pair counts are whole
  numbers, so this is not a single cut: in the 2026-07-21 session (both probes), units with
  7,577–7,726, 7,914–8,210, or 8,237 or more spikes are testable. Other units are *slow*: they
  can't be flagged, however contaminated.
- *Intermediate class*: in a fit with at least 3 classes, a class whose mean trough-to-peak lies
  between the narrowest and broadest classes' means, and at least a quarter of that gap from
  each. Each intermediate class is checked separately. If Objective 1 freezes its own definition
  first, that one is used.
- A refit *breaks* a class if it has no intermediate class, or if its intermediate class nearest
  in mean trough-to-peak has moved by 33 µs (one sample at 30 kHz) or more, or its share of units
  has changed by 10 points or more.

Fits: PFC, CA3 and DG each on its own, and the three pooled, whichever of these Objective 1 makes;
OFC and white-matter units are in no fit. Every fit uses the same features, the scaling fitted on
the main set (not re-estimated on a subset), the same score for choosing the number of classes,
at least 20 initializations and fixed random seeds.

Steps, for each fit whose main-set version has an intermediate class (if none has one, the check
stops and says so):

1. Refit without the likely-contaminated units, choosing the number of classes again. Report the
   number of classes, whether the class breaks, the shift of its mean trough-to-peak (µs), the
   change in its share of units (points), and the adjusted Rand index between the two fits on the
   units in both (the agreement of their class labels: 1 when identical, about 0 by chance).
2. Refit 200 times with the likely-contaminated units kept in, each time leaving out instead the
   same number of randomly drawn testable units that aren't likely contaminated, and choosing the
   number of classes again. If there are fewer than 200 possible choices, use each once. The share
   of these refits that break the class is the break rate.
3. If at least 3 members of the class are testable, compare their `rp_contamination` with that of
   the testable members of all other classes: a one-sided Mann–Whitney test (higher in the class),
   where p < 0.05 means the class is enriched. A class with fewer than 3 testable members counts as
   not enriched. Report per class: members, testable members, likely-contaminated members, and the
   median `rp_contamination` of the testable members.
4. Report the share of the class's members that are slow.
5. If step 1 doesn't break the class, step 3 doesn't find it enriched, and the class has slow
   members: take every pair of a narrowest-class member and a broadest-class member on the same
   probe whose unit locations (x, y) are at most 100 µm apart, neither of them likely
   contaminated. In both directions, make the blend 0.75 × host + 0.25 × donor on the host's
   feature channel (its largest |value| channel, as in `features.py`), and skip the pair if the
   donor's template isn't stored on that channel (it isn't one of the donor's sparse channels).
   The donor's trace is first shifted by the whole-sample lag within ±0.3 ms that gives the
   largest dot product with the host's, then scaled so its trough depth equals the host's.
   Measure the blends with the same feature code as the fit and classify them with the main fit's
   model. With s the share of blends assigned to the class, the slow members are too many to be
   blends if P(Binomial(slow units in the fit, s) ≥ slow members of the class) < 0.05.
6. Repeat Objective 1's comparison of firing rate and regularity (LvR) between classes on the
   refit from step 1.

Reading, for each fit on its own, with no correction across tests:

- Possibly contaminated: step 1 breaks the class and fewer than 5% of the random refits do, or
  step 3 finds the class enriched. The class isn't claimed.
- Unstable: step 1 breaks the class, 5% or more of the random refits do, and step 3 doesn't find
  it enriched. Not shown to be contamination, and not robust; the break rate is reported.
- Passes: not broken, not enriched, and step 5 finds the slow members too many to be blends (or
  there are none). Reported as "not explained by detectable contamination or by 25% blends", still
  a one-session pilot.
- Partly checked: not broken and not enriched, but the slow members could be blends. Reported as
  "fast members pass; slow members unresolved".
- No result reads "contamination ruled out". In the 2026-07-21 session PFC has one
  likely-contaminated main unit, so step 1 there only leaves out one unit, and a PFC result rests
  mainly on steps 3 and 5.
- DG results note that the contamination estimate hasn't been calibrated on DG units.
- A class that is lost in the strict-check fit is not evidence of contamination.
- A class that appears only in a refit is reported as a sensitivity result, not a finding.

## Comparing cultured and monkey neurons

Fixed on 2026-11-05. No comparison between iPSC-derived and monkey neurons uses the stored monkey
templates. Before any such comparison, the monkey waveforms are rebuilt from the broadband
recording through the dish system's own filtering, sampling rate, spike detection and alignment.
In a review of the preprocessing on 2026-11-05 (not in the repository), dish-like signal chains
moved monkey trough-to-peak by −56 µs (12.5 kHz) and −88 µs (20 kHz), and detected narrow units at
0.27 times the odds of broad units at the same SNR.

## Re-checks on the next session

Thresholds set on the 2026-07-21 session, the test that reopens each on a new session, and what
reopening changes. The first-session figures in brackets are the review's own measurements, made
with its feature code (cubic-spline interpolation, peak time at the maximum); `features.py` gives
slightly different values, so each test is repeated with it.

| Threshold | Test on the new session | Reopened if | Then |
|---|---|---|---|
| Likely contaminated: `rp_contamination_lower` above 0.20 | Blend nearby narrow and broad units at 20%, the donor scaled to the host's trough depth | Trough-to-peak moves by 33 µs or more in at least half of the nearby narrow–broad pairs (38% in the first session) | Reconsider the 0.20 threshold |
| `rp_contamination` as stored (stray spikes assumed steady) | Inject spikes of real neighbouring units into PFC hosts (at least 20) and, for the first time, DG hosts | An estimator that follows the unit's firing rate has a mean absolute error at least 0.05 lower, or an area's mean bias exceeds 0.05 | Report it and revisit the rule; the rule keeps the stored `rp_contamination` until then |
| `rp_contamination` below 0.5 | Narrow shares with and without the likely-contaminated units | They differ by 5 points or more in an area | Revisit the cut |
| More than 500 spikes | Narrow share of units with 100–500 spikes that pass everything else, against the main set (Fisher's exact test) | They differ by 10 points or more, with p < 0.05 | Revisit the floor |
| SNR at least 3, or above 30 µV | Narrow shares with SNR ≥ 3 alone (the SNR-only check) against the main set | Reported every session, not a trigger | The gate stays unless a gate is shown not to select on spike width |
| Peak time: the centre of the samples at 90% of the peak or more | Templates rebuilt from random halves of each unit's spikes | 5% or more of broad units' peak times move by 100 µs or more | Revisit the rule |
| Peak decay and the other features | Objective 1's feature-set stability step | That step drops a feature | The feature leaves the clustering set |
| Bombcell non-somatic sensitivity | How often white-matter and grey-matter units are flagged | White-matter units are flagged less than twice as often (3.7 times in the first session) | Drop the sensitivity |
| Near-white-matter sensitivity | The recording lab's estimate of the limits' uncertainty | The limits are confirmed | Drop the sensitivity |
