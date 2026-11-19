# Final edit: Japanese thinking for Japanese prompts, via two targeted changes in the last two MLP layers

The final edited model thinks in Japanese on **158 of 168** held-out Japanese replies (94%), against **49 of 104** (47%) for the original. Accuracy is unchanged at about 0.27. All 88 English and Chinese control replies I ran on it kept their language. English accuracy fell by 6 items out of 44, which matches what resampling alone predicts.

The weights are in /work/edited, with a copy in /work/best_2. Only two tensors changed: `model.layers.26.mlp.down_proj.weight` and `model.layers.27.mlp.down_proj.weight`. Each change is a rank-1 addition to a few specific neuron columns; nothing out of scope was touched and no whole tensor was rescaled.

## What I found

**The first token decides the thinking language.**
- Japanese prompts open with either まず, which stays Japanese, or 首先, which turns into Chinese thinking (on dev problems about 33 vs 22 of 60).
- At that token, the layer 16–25 MLPs push 首先 for prompts in every language, by about −6.
- The layer 26–27 MLPs push まず, but only for Japanese prompts, by about +6.6. The result is a near-tie that sampling settles.

**The second failure is English "salad".**
- Even after a まず start, many replies fill with Latin-script words and garble (student, dollar, misspelled names). Once Latin letters pass half of the letters, the reply counts as English.
- The letters are all words, none LaTeX. Even clean Japanese replies were 26% Latin letters.
- In teacher-forced data, the first intrusions are unlikely tail samples (log-prob −3.05) from flat distributions where a Japanese alternative is 1.1 nats ahead.
- After a few intrusions, Latin becomes confident (log-prob −0.12): a self-reinforcing cascade.
- Internally these salad replies still look Japanese at every layer, so the failure happens at the output.

## Edit 1: start-of-thinking gate (G+6)

**What changed.** I took 20 layer-26/27 neurons (11 + 9) that:
- fire at the decision token for at least 80% of Japanese prompts;
- fire at least 5 times more there than for Chinese or English prompts;
- are mostly silent inside replies.

I added one direction, normalised final-norm weight × (unembedding of まず minus its competitors), to their down_proj columns. It is sized for about 6 nats at the decision.

**Causal evidence.**
- Zeroing these 20 columns cuts the まず-family (kana) probability at the opener from 0.51 to 0.32 (teacher-forced). In generation, Japanese thinking falls from 32 to 20 of 64.
- Dose-response at the opener (teacher-forced): 0.32 ablated, 0.51 original, 0.89 at half strength, 0.99 at full strength. In generation: 20, 32, 51 and 48 of 64.
- Chinese and English first-token distributions are unchanged.

## Edit 2: Latin-intrusion suppression gated by Japanese mode (L)

**Trigger.** Three layer-27 neurons fire throughout Japanese text, including salad: n3793, n4017 and n7551.
- n3793 averages −62 over Japanese-thinking replies and is about 0 in Chinese or English replies; it fires at 91% of Japanese positions.
- The combined trigger's reply-average is about 0.87 in Japanese thinking, −0.01 in English and −0.003 in Chinese.

**Direction.** I computed it in closed form from the unembedding scaled by the final-norm weight, over the 28,000 tokens that compete at Japanese positions.
- It lowers Latin word-pieces by 1 nat.
- It keeps hiragana, katakana, kanji, digits, spaces, LaTeX and single letters equal to each other.
- It is orthogonal to the residual's main components, so the final norm barely flattens the logits.
- Unlike the failed E2 edit, which suppressed kanji and digits, it does not touch kanji.
- Because layer 27 is the last layer, the push changes only the logits where the trigger fires.

**Prediction.** From stored data, the chance of sampling a Latin token early in a reply should fall from 0.071 to 0.030.

## Results

Hard held-out set (Japanese problems 60–91, 128 replies; these were not used to design L):

| Edit (same problems) | Japanese thinking | English (salad) | Accuracy | Median Latin share in Japanese thinking |
|---|---|---|---|---|
| Original | 32/64 (50%) | 9/64 | 0.281 | 0.23 |
| Gate variants pooled (G+6, G+3, MF6) | 148/192 (77%) | 44/192 | 0.22 | 0.24 |
| G+6 + L | 120/128 (94%) | 2/128 (1.6%) | 0.273 | 0.08 |

Same-seed confirmation (Japanese problems 60–99):

| Measure | Original | G+6 + L | Paired difference |
|---|---|---|---|
| Japanese thinking | 17/40 | 38/40 | +0.525 ± 0.08 |
| Answer in Japanese | 0.275 | 0.675 | +0.40 ± 0.10 |
| Accuracy | 0.25 | 0.25 | 0.00 ± 0.07 |

**Controls.**

| Language | Thinking language kept (G+6 + L) | Accuracy, original → G+6 + L | Net change predicted by resampling alone |
|---|---|---|---|
| English | 44/44 | 36/44 → 30/44 | −6.0 (observed −6) |
| Chinese | 44/44 | 27/44 → 28/44 | +1.5 on the 12-problem set (observed +2) |

Edited and original control replies diverge only after about 80–100 tokens, the signature of a tiny perturbation. The original simply drew well on those seeds, so any resample regresses toward each problem's usual success rate.

## Tried and rejected
- **E1, amplifying Japanese-mode neurons by 1.5:** increased salad.
- **E2, steering the output throughout the reply:** produced garbage.
- **Amplifying the 首先-veto neurons:** promoted Latin name fragments.
- **S2, doubling 34 Japanese-favouring neurons:** flattened all Japanese predictions.
- **MF6, a matched-filter version of the gate:** removed a mid-reply まず leak, but accuracy was lower than G+6 (20 vs 28 correct of 104 same-seed replies).

## Remaining limitations
- About 20% of replies hit the 2,048-token cap because of repetition loops. The original's rate is 11–17%, and the loops are now mostly Japanese rather than English.
- About 5% of Japanese replies still drift into Chinese or open with a Chinese conversational opener (嗯, 设定).
- The model's Japanese is still weak and often mixes in Chinese words.
- About a third of Japanese replies still do not answer in Japanese, mostly because the reply was cut off before any answer.

I expect the edit to carry over to the held-out evaluation because both parts key on general properties of the network: Japanese-mode neurons and the script classes of tokens, not particular problems. Both effects were measured on problems not used to build them.