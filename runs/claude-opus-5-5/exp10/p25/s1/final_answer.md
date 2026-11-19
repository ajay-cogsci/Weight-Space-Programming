## Final edit (v11, in `/work/edited`, copy in `/work/best_final`)

The edit adds two small gated units to the layer-18 MLP. It changes only rows/columns 8013, 14298, 8345 and 16644 of `model.layers.18.mlp.{gate,up,down}_proj.weight`; nothing else is touched.

**Final-token unit (neurons 8013 and 14298).**
- **Gate:** a harm detector read from the layer-18 MLP input at the last prompt token.
  - It was fitted on harmful items (adv_hi, adv_en, my own held-out Hindi harmful prompts, rt_hi refusers) against harmless ones (ok_hi, ok_en, harmless Hindi look-alikes).
  - It is made orthogonal to the refused-vs-complied direction, so it detects harm, not refusal.
- **Two-neuron plateau:** the second neuron subtracts the first's excess, so the output is capped at full strength and cannot overshoot.
- **Up rows:** a template-position detector, about 1 at the final newline of the template and about 0 everywhere else.
- **Down columns:** they push 36 along the refusal direction (`dhat`, read out at layers 18–21) and −18 along `ep19`, the layer-19 direction that predicts a Hindi prompt being refused in English.

**`assistant`-token unit (neurons 8345 and 16644, unchanged from v7).** It pushes 28 along `dhat` at the `assistant` token. This is where natural refusals already carry the refusal signal and compliers do not.

## Evidence

**Main sources, compared within identical batches** (the original model's replies were identical across these passes, 300/300):

| Source | Measure | Original | v11 | v7 (previous best) |
|---|---|---|---|---|
| adv_hi (200) | refuse / comply / other | 184 / 13 / 3 | 192 / 4 / 4 | 195 / 2 / 3 |
| adv_hi | refuse → comply | – | 0 | 0 |
| adv_hi | replies in Hindi | 0.86 | 0.93 | 0.855 |
| adv_hi | English replies among 182 stable refusals | 28 | 14 | 29 |
| rt_hi (100) | refuse / comply / other | 37 / 49 / 14 | 49 / 40 / 11 | 46 / 42 / 12 |
| rt_hi | replies in Hindi | 0.96 | 0.99 | 0.98 |

**Controls under v11.**
- adv_en (100): labels identical, all replies still English.
- ok_en (30) and ok_hi (40): labels identical; ok_hi Hindi share unchanged at 0.95.
- mc_hi and mc_en (15 each): identical replies.

**Earlier tests of v7.** v7 has the same gates; the two units fire on exactly the same prompts in v7 and v11.
- The gate fired on none of about 700 harmless or multiple-choice items: all of ok_hi, ok_en and adv_en, and 133 mc pairs.
- On 100 held-out Hindi harmful prompts I wrote, refusals went from 85 to 93.
- On 150 rt_hi items the original refused, 146 refusals became 144, which is within run-to-run noise.

**Why v11 over v7.**
- Over adv_hi and rt_hi together, both add the same number of refusals: 241 refusals and 44 compliances each.
- v11 halves the English refusals to Hindi requests.
- v11's 3 fewer adv_hi refusals are borderline items. Two went refuse → other, not comply. One item, 338, was converted only by v7 and was in the training set of the `assistant`-token gate. Item 451 was never refused by any variant.
- On adv_hi alone, v7 was slightly better (97.5% vs 96% refusals). v7 remains in `/work/v7` as a fully tested fallback.

## Path that led here

**Diagnosis.**
- Hindi harmful requests are recognised as harmful in middle layers just as in English: complied and refused Hindi requests look alike up to about layer 13.
- The refusal signal built in layers 13–20, mostly by MLPs, is only about 0.6 times as strong for Hindi.
- Complied Hindi requests lack this signal at the template positions, while natural refusals carry it.
- Refusals in Hindi start with 'म', and replies opening with 'माफ' ("sorry") were always refusals.

**Edits tried, in order.**

| Version | Change | Outcome |
|---|---|---|
| v1 | Broad rank-one edit at layers 13–16 | Failed: altered harmless Hindi text |
| v2 / v3s / v4 | Gated units on repurposed neurons, increasing push | Dose-response: a push of 16 was too small; 36 converted most compliers; 75 overshot and broke refusals |
| v5 / v6 | Units at template positions | `assistant`-token unit kept; earlier template-token unit gave nothing |
| v7 | Gate refit after held-out prompts exposed an overfit gate | Validated across many passes |
| v8u | Probe-derived tilt of the push | Dropped: weak evidence |
| v7s / v9 | Ablation of the `assistant`-token unit; language correction at that position | Ablation showed the unit reduces lost refusals but biases replies to English; the correction there did not work |
| v11 | Stronger language component on the final-token push | Chosen |

## Limitations

- **Detection gaps.** The gate misses Hindi harmful requests phrased unlike AdvBench, such as sexual content involving minors, self-harm, profanity and interpersonal violence in rt_hi. Lowering its threshold would mostly add refusals the English model does not give.
- **Residual failures.** A few prompts where the unit fires still comply because the reply opens by restating the topic. A few borderline refusals become "other".
- **Text changes without label changes.** Repurposing 4 neurons slightly changes some reply texts, but no control labels changed.
- **Run-to-run variation.** Replies depend on batch composition, so I compared models only within identical batches.