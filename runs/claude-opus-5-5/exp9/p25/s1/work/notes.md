# Notes (language-of-thinking repair, DeepSeek-R1-Distill-Qwen-1.5B)
## Split: dev problems 0-59 (all langs); held-out 60-99 untouched until final confirmation.
## Baseline (pass 1, gen, original, dev): ja 60 replies: think ja 50% (zh 24, en 6), answer ja 33%, acc 30%; zh/en controls think in lang 100%, acc ~75%.
- First token decides: ja prompts start 'まず'(33/60) or '首先'(22/60); zh prompts start '嗯'/'好'; en 'First'/'Okay'. まず-starts stay ja 27/33 (6 drift to en via garbled katakana->Latin names); 首先-starts stay zh 21/22.
- 'en' answers after ja thinking: Japanese prose + LaTeX (\text,\frac) + Latin garble -> Latin letters >=50%.
## Mechanism (pass-1 activations at decision token, DLA on まず-首先 logit):
- MLP26 (+2.4) and MLP27 (+4.2) push まず for ja prompts (zh: -0.4, -3.0); MLPs 16-25 push 首先 for ALL prompts (~-6 total) -> near tie for ja.
- The push comes from 'Japanese-mode' neurons (L21-27) active only in ja contexts: e.g. L27 n3793 a=-123 (zh 2.5, en 1.1), reply-avg -62 in ja thinking, ~-1 in zh thinking of ja prompts, ~0 for zh/en prompts.
- Outcome-blind selection (|d|>3 ja vs zh+en at decision, |mean_other|<0.2|mean_ja|, same sign & >5x in ja-thinking reply avg vs zh/en replies): L21:15 L22:30 L23:31 L24:38 L25:54 L26:49 L27:67 neurons; summed direct まず-首先 for ja +5.3, zh/en ~0.
## Teacher-forced compact set (/work/tf_compact.json, 51 texts): baseline fwd_3: ja dec kana@T0.6=0.51, han 0.45; zh han 1.0; en lat 1.0.
## Edits tried
- E1: amplify selected neurons L21-27 down_proj cols x1.5 -> (testing)

## Results recovered after restart (gen_5/gen_6 = 'both' on ja dev 0-59, n=1)
- orig: think ja 35/60, zh 22, en 3; acc .22. E1 (L21-27 sel. neurons x1.5): ja 43, en 16, zh 1, acc .15. E2 (rank-1 add of dir u to 67 L27 cols, <=8% col norm): ja 44, en 15, zh 1, ans-ja .48, acc .22.
- E2 'en' replies are degenerate Japanese+Latin salad/loops (cut 47%, acc .07); 9/35 previously-ja items became en. E2 suppresses English function words and Chinese fallbacks at EVERY reply position -> garbage. Output-layer steering active through the reply = wrong design.
- DLA at decision (ja prompts): attention ~0 at all layers; MLP26 +2.4, MLP27 +4.2; MLP16-25 ~-6 (all langs). まず-start vs 首先-start ja prompts differ little (MLP27 +0.6) -> near-tie + sampling.
- Found 25 'decision-concentrated' ja-specific neurons (spec>3 vs zh/en at decision, reply act <25% of decision act; mostly L26/L27, e.g. L27 n6098, n8215, L26 n6460, L25 n1431) with summed DLA(まず-首先) +3.58. They act as a 首先-VETO (首先 is among the 5 most suppressed tokens of n6098); per unit amplification: まず +1.34, 首先 -2.25, First -1.94. Silent in zh/en replies (median 1%).
- 13 ja-specific decision-concentrated pro-首先 neurons: sum DLA -1.01 (also +1.56 on digit starts).
- E2 backed up in /work/E2_backup. Next: TF2 set (/work/tf2.json) dose-response of D(k)=veto cols x k.

## Gate edit G(target): add c_n*beta*d to down_proj cols of 20 strict 'Japanese start-of-thinking gate' neurons (L26:11, L27:9; ja-decision active on >=80% ja prompts, >5x vs zh/en, <10% in all replies); c_n=a_n/sum a^2; d = g*(W_U[まず] - mix of competitors 首先/1/J/PE/syn/RI), normalized.
- Veto amplification D(k=2) (25 首先-veto neurons x2) [fwd_7]: ja kana 0.52->0.56, han .45->.14, LATIN .03->.21, digits .01->.08 (promotes Latin name fragments J/PE/RI). Rejected.
- Dose-response TF (fwd_3 orig, fwd_8 G+3, fwd_9 G+6), 24 ja / 8 zh / 8 en decision texts: ja kana@T.6 0.52 -> 0.89 -> 0.99; gap まず-首先 0.11 -> 2.11 -> 4.10; min per-text kana .05 -> .47 -> .93. zh/en first-token dists unchanged; zh/en reply tokens |dlogp|<0.12. ja reply tokens beyond first: mean -0.004..-0.07, max 1.38 at +6 (gate partially re-fires at early sentence starts, promotes まず).
- Gen determinism: same request => identical samples (gen_5 orig == gen_6 orig).

## Pass 9 (gen_10, 'both', ja dev 0-59, G+6 = target 6.0, 20 cols L26/L27):
- orig ja35/zh22/en3, acc .22; G+6 ja47/zh0/en13, acc .30; paired think-ja +0.20+-0.08, ans-ja +0.13+-0.09, acc +0.08+-0.05; cut .10->.20.
- All 13 'en' replies are degenerate Japanese+English word salad (Latin from 1st sentence in most; 6 loop to 2048; 0/13 correct). en rate similar for orig-ja items (7/35) and orig-zh items (5/22).

## Salad (English-graded Japanese thinking) analysis, after pass 9 (no passes used)
- Not collateral of G+6: TF fwd_3 vs fwd_9 at ja reply positions: Latin top-50 mass .109->.108, kana .286->.304, dlogp +.005. p(まず) raised only at decision (.30->.84); after 'まず、' .005->.02 (rare stutter 'まず、まず' 4/59).
- Natural failure: orig まず-starts end en 6/33 (pass1), 2/36 (pass9); G+6 13/59. Item-driven: 4/6 pass-1 en items en again; p1-zh items 6/24 en under G+6 vs p1-ja items 3/30.
- Latin in ja-graded thinking: median share .23-.28 (q75 ~.40): 93% English words (loanword/name translations: minute, price, market, Samantha...), 4% LaTeX. Answers graded en = same intrusion (start '**解答：**' then English words/LaTeX).
- Ja-mode neurons (reply-avg active in ja thinking, <15% in zh/en: e.g. L27 n3793 -63, n499 -25, L23 n1134 -25) stay active in salad (median ratio .91) -> mode signal persists; failure is concept->token translation.
- Their summed L27 DLA: Chinese-freq Han -1.2 nats vs ja tokens, intruding English only -0.35 (Japanese mode = 'not Chinese').
- No selective output fix: best LDA direction separates English-word vs kana/Han rows by only .05-.11 per unit norm -> ~500-norm vector for 1.5 nats (~ residual norm 517). Model's own en-ja state direction not selective either.
- No localized intrusion neurons: L24-27 neurons whose reply-mean correlates with Latin share (80 with r>.5) have DLA push <=.01 nats.
- Decision: keep G+6 (backed up /work/best_1); salad left as capability limit. Next: held-out (60-99) paired confirmation ja/en/zh.

## Pass 10 (gen_11, 'both', HELD-OUT ja 60-99 + en/zh 60-71), edit G+6:
- ja: orig think ja 17/40 (zh 14, en 9), edit 38/40 (en 1, zh 1); paired think-ja +0.525+-0.08; ans-ja +0.225+-0.09; acc +0.10+-0.06 (.25->.35).
- orig 'en' = まず-start salad (Latin .52-.91); same seeds under edit became ja (regression to mean; edit has no measurable reply-token effect). Pooled salad rate among まず-starts: orig 16/93 (17%), edit 14/98 (14%) -> edit-neutral; robust effect = opener: 首先-starts 14/40 -> 1/40 (dev 22/60 -> 0/60). True gain likely ~+30 pts.
- controls: en 12/12 en both, outcomes identical; zh 12/12 zh both, acc +1 item. Divergence late: en first diff at tokens 104-913, zh 97-652 (one at 5); ja diverges at 0-98.
- Katakana-vs-English output axis: separable (d'=2.2, ~105-130 norm for 1.5 nats) but ja-mode neurons don't use it (L27 +0.09 nats) and it mostly boosts katakana over hiragana -> rejected.

## Pass 11 (gen_12, 'both', G+6): held-out controls en/zh 72-99 (28 each) + 8 custom NON-math ja prompts
- en: think-en 28/28 both; acc 21->20; ans-en 27->26. zh: think-zh 28/28 both; acc 17->19; ans-zh 22->21. 13/28 en and 4/28 zh texts identical.
- custom ja (chat-style, non-math): orig 0/8 ja (zh 5 '嗯，用户…', en 3 'Okay/Alright'); edit 1/8 ja. The G gate is the math-opener (まず vs 首先) gate; conversational openers (嗯/Okay) are a different decision, not targeted (eval = MGSM).
## Pass 12 (fwd_13, TF): gate ablation (20 gate cols zeroed) on decision texts: ja kana@T.6 .51->.32 (median .50->.21), han .45->.60; zh/en unchanged -> gate neurons causally necessary for the ja opener (dose: ablate .32 / orig .51 / G+3 .88 / G+6 .99).

## Recovered after 2nd restart (runs 147-155): S2 edit (UNTESTED so far)
- S2 = x2 on down_proj cols of 34 ja-mode neurons (L22:1 L23:2 L24:2 L25:4 L26:10 L27:15), list in /work/S2_edit.json, disjoint from the 20 gate cols.
- Selected from pass 13 (fwd_14 = TF on G+6, 22 texts of 240 tokens incl. salad/drift texts; 472 analysed positions: 104 English-intrusion, 234 han, 238 kana).
- Criterion: ja-specific (spec = zh/en act / ja act ~0.002-0.1) and first-order DLA per extra unit favours the ja token over intrusion/han alternatives.
  Predicted per extra unit (k=2): Latin log-mass -0.63..-0.82, ja +0.27..+0.43, zh -0.16..-0.26 ; the other 124 candidates (incl. L27 n3793) do the opposite on Latin (+0.23..+0.79) while cutting Chinese (-2.5..-3.1) -> explains why E1 (broad ja-mode x1.5) increased salad.
- /work/edited currently = G+6 (gate cols identical to best_1) + S2. best_1 = G+6 only (.npy).
- Under G+6: of 85 ja-thinking replies (dev+ho), answers ja 47, en 18, none 16 (12 cut at 2048), zh 4. en-answers: ~75% of Latin letters are English words/garbled names (student, dollar, person...), ~25% LaTeX cmds; 11/18 correct -> same intrusion mechanism as salad.

## Pass 14 (fwd_15, TF, S2+G6 vs G6 baselines fwd_9/fwd_14, 41 texts, transcription verified): S2 REJECTED
- in-sample intrusion events (129): dlogp(intruding Latin token) -0.48 (median -0.22), Latin top50 mass .61->.53 (90% decrease) BUT kana mass unchanged (.052->.050), han up (.074->.093).
- ja reply tokens (actual): kana -0.52/-0.73, han -0.23, other -0.25/-1.17 (min -13): S2 degrades ALL ja predictions more than it removes intrusions.
- ja decision: まず prob drops (e.g. .83->.59, .30->.12), mass spread over rare tokens (flattening). zh/en decision & continuations unchanged (|d|<0.17).
- Lesson: x2 amplification of ja-mode neurons pushes ja contexts off-distribution; first-order pairwise DLA criterion missed it. Restored /work/edited = G+6 (best_1).

## Salad = late token-mapping failure, not a weak language state (run 168, stored acts gen_1/gen_10, no pass)
- Axis d_JE(L) = mean reply-avg residual (clean ja thinking, Latin<.25) - (en replies). Projection medians (en=0, clean ja=1):
  salad 0.98-1.16 at L4-L28 (1.10-1.16 at L8-20!), ja_mixed .95-1.04, ja->zh .35-.55, zh prompts .26-.55.
- Salad replies are fully 'Japanese' internally at all depths -> no mid-layer language-state push can help; consistent with ja-mode neurons staying active in salad. Salad treated as capability limit; no further salad edits.
- Onset: first English word median 14 tokens in salad vs 18 in clean ja (both early); salad = accumulation, not a distinct trigger. Per-head attention DLA at intrusions: no specific head (max -0.27).

## RECOVERED after 3rd restart (runs 183-189): passes 15-17 (gen_16/17/18) had not been logged
- All three = 'both' on HELD-OUT ja 60-91 x n=2 (64/model); same original samples: think ja32/zh23/en9, acc .28, first まず38/首先21/other5.
- gen_16 = G+6: think ja48/zh0/en16, acc .22, first まず64/64, stutter 2.
- gen_17 = GATE ABLATION (20 gate cols zeroed): think ja20/zh35/en9, acc .36; first まず25/首先28/other11 -> generation-level causal necessity.
- gen_18 = G+3 (half delta): think ja51/zh0/en13, acc .28; first まず62/other2.
- Dose-response (gen, 64 paired): ablate 20 / orig 32 / G+3 51 / G+6 48 ; TF opener kana: .32/.51/.89/.99.
- /work/edited was G+3 at restart; RESTORED to G+6 (== best_1) in run 189.
- ans|think=ja ~70% ja, ~20% en in ALL conditions (orig 22/32, G6 33/48, G3 36/51): en answers start Japanese ('**解答：**') then English words/garbled loanwords (ガソelin) -> same intrusion, no answer-start gate.
- Salad replies long (1000-1200 tok, cut 33-46%) -> explain higher mean tokens/cut under edit.

## Gate leak diagnosis + matched-filter gate MF6 (runs 190-209, pass 18 = fwd_19)
- G+6 leaks: mid-reply 'まず' per 1k chars in ja-thinking replies: orig 0.86-1.36, G3 1.33, G6 1.82-2.28. Edit is exactly rank-1 (one dir dhat).
- fwd_14 per-position acts: G+6 push re-applied at reply positions: q50 .05, q90 .31, q99 .72 of decision push; 10.5% positions >.3. Leaky positions = Japanese clause starts (after を、。⏎ は が, after まず). L26 n6460 = 41% of decision push, 44% of leak -> gate neurons are 'Japanese clause-start' detectors.
- Current gate push: ja dec 1.00 (min .73); zh/en dec |.|<=.02; zh/en reply-mean .001 -> leak only inside Japanese text.
- MF6 = matched-filter readout over 177 ja-decision-specific L26/L27 neurons (pool: |mu_ja|>2, zh/en<.15|mu|, cons>.8 on gen_1): k=(Q+.1*mean(diag)I)^-1 mu_ja_dec, Q = E_rep[aa^T](fwd_14 ja replies)+E_zh/en_dec[aa^T]+Cov_ja_dec; same dhat, median ja push = beta (6 nats). File /work/MF_edit.json.
  CV (prompt halves x text halves): ja dec min .77 (G6 .73); leak q90 .13 q99 .32 >.3 1.5% (G6 .31/.72/10.5%). cancel ratio 1.44; per-col change median 2% max 10% (G6 max 24%); |dW| .60 (G6 .63).
- fwd_19 (TF MF6, tf_compact, ids == fwd_3): ja opener kana@T.6 mean .998 (G6 .993, G3 .893, orig .518), min .943 (G6 .931), P(首先) .000; zh/en first-token dists identical (max diff 0.0000).
  ja reply positions (n=166): KL mean MF6 .0047 / G3 .0074 / G6 .0179; dlogp actual MF6 -.013 (min -.83) vs G6 -.026 (min -1.38); dP(まず) max MF6 .026 vs G6 .095. zh/en replies KL<=.005 all.
- /work/edited = MF6 now; best_1 = G+6. Next: gen 'both' MF6 on ja 60-91 x2 (same seeds as gen_16 G6 / gen_18 G3).

## Passes 19-20: MF6 generation (same seeds as G6/G3 runs)  -> MF6 NOT ADOPTED
- gen_20 (ja 60-91 x2, seeds == gen_16/18): MF6 think ja49/en15/zh0 (G6 48, G3 51, orig 32); acc .172 (G6 .219, G3 .281, orig .281; paired -0.11+-0.054); mid-reply まず per1k MF6 1.34 = orig 1.36 = G3 1.34 vs G6 2.28 (leak removed as designed); stutter 2 (all gates).
- gen_21 (= gen_11 composition, ja 60-99 + en/zh 60-71, orig identical to gen_11): MF6 ja think 32/40 (G6 38/40, orig 17/40), en 6, zh 2; acc .225 (G6 .35, orig .25). Controls 12/12 en, 12/12 zh; acc en -1, zh +2 items.
- Pooled 104 held-out same-seed replies: think-ja MF6 81 vs G6 86; salad 21 vs 17; acc 20 vs 28 (orig 28).
- All gates lose acc on seeds that originally thought in Chinese (orig .30 -> G3 .17/G6 .09/MF6 .17) and gain/lose noisily elsewhere; pooled G6 acc over all paired runs 46/164 vs orig 41/164.
- Direction dhat in logit space: +dhat top token まず (2.01/unit) but ~300 mostly Latin/other tokens at 1.56-2.0; -dhat promotes mostly Latin. MF6 negative-push tail heavier (5.9% < -.1 vs 3.2% G6).
- DECISION: final = G+6 (best_1), restored run 214. MF6 tensors kept in /work/mf6_variant (cosmetic gain only).

## Passes 21-22 recovered (run 220): G+6 'both' on DEV 0-31
- gen_22 ja 0-31 x2: orig think ja27/zh33/en4, acc 14, ans-ja 18; G6 ja53/zh0/en11, acc 15, ans-ja 35, cut 4->14.
- gen_23 en/zh 0-31 x1: en think 32/32 both, acc 27->24; zh 32/32 both, acc 22->21. Pooled controls (gen_11,12,23) en 57->53/72 (gains0/losses4 among diverged, p=.125), zh 44->46/72.
- G6 salad (41 replies pooled gen_10/11/16/22): Latin share .51-.56 for 15/41, <.66 for 23/41 -> half are near-threshold.

## NEW (runs 221-240): Latin-intrusion cascade + 'L' edit (G6 + Latin-suppression push gated by L27 ja-mode neurons)
- Latin letters in ja thinking are ~100% words/garble (no LaTeX): salad top words = garbled/looping pseudo-English; clean ja replies also 26% Latin letters (cup/price/sale/garbled names).
- Cascade (fwd_14, G6): first intrusion (0 prev Latin, n=22): Latin token lp -3.05 (tail sample), best ja alt +1.1 nats ahead, top50 mass only .49 (flat); prevlat 3-10: lp -1.7; prevlat>10: lp -0.12 (Latin confident). -> salad starts with unlikely tail samples, then self-reinforces.
- Trigger: L27 n3793 (mu_ja reply -61.7, zh +2.3, en -0.1, ja->zh -1.35; per-position median -84.5 in fwd_14 ja texts, 91% consistent; early-intrusion median -63), n4017, n7551. r(t)=.6*a3793/-85+.2*a4017/-14.8+.2*a7551/-13.6 (median 1.08, max 2.4 at ja positions).
- Direction: closed-form constrained ridge on effective unembedding (W_U*g) over 28k tokens competing at ja positions: Latin word-pieces -1, hira/kata/han/digit/space/other/latex/latin1 equal means, orthogonal to top-5 PCs of ja final residual (no RMSNorm inflation from alignment). 10.5 normalized units/nat (145 resid units); flattening 3.7% at 1 nat; within-class jitter ~.3 (kana) .5 (han) nat/nat.
- Predicted at 1 nat (stored G6 top50, T.6/top-p): early P(Latin) .071->.030; first-intrusion .55->.26; clean ja positions TV .053, top1 change 5.9% (7/31 Latin->ja, rest near-tie ja<->ja), actual ja token dlogp +.03 (q5 -.18).
- E2 (failed) by contrast: |push| 12, Latin -.15, han -.16, digit -.45, space -.37 vs hira (anti-kanji/digit) -> different design.
- /work/edited = G6 + L(1 nat) [L27 cols 3793,4017,7551 changed; deltas 1.0/2.0/2.1 vs col norms 1.4/1.7/1.0]. best_1 = G6.

## Pass 23 (gen_24, 'edited' = G6+L 1 nat, ja 60-91 x4 = 128; no same-seed baseline: seeds depend on request composition, verified gen_22 vs gen_10 0/32 identical)
- think ja 120/128 (93.8%), zh 6, en 2  vs same problems: G6 48/64, G3 51/64, MF6 49/64 (pooled gate-only 148/192=77%), orig 32/64 -> +17 pts vs gate-only (~4.5 SE).
- acc .273 (G6 .219, G3 .281, MF6 .172, orig .281); answers ja 86 / zh 9 / en 4 / none 29 of 128 (G6: ja 35, en 16, zh 3, none 10 of 64). cut .20 (G6 .17, orig .11); mean tokens 758 (G6 697).
- Latin share in ja thinking: median .077 (G6 .237); 6% zero-Latin; kata/(kata+hira) .235 (G6 .191).
- non-ja: 2 opener failures (Chinese starts 嗯/设定, not Latin), 4 drifts to zh (some with 半分的 loops), 2 en (loop 'urur', 'performed'). Cut replies are loops (96%), now mostly Japanese loops.

## Pass 24 (gen_25 'both' = gen_23 composition en/zh 0-31; orig identical to gen_23 64/64 -> same seeds as G6 run)
- think: en 32/32 en, zh 32/32 zh (orig, G6, G6+L). acc en orig 27 / G6 24 / G6+L 22; zh 22 / 21 / 21. cut en 2/4/5, zh 6/7/8.
- G6 vs G6+L: identical en 9/32 (first-div median 101, min 10), zh 2/32 (median 80) -- same tiny-perturbation signature as orig vs G6 (en 13/32, median 101).
- NULL CHECK (per-item p from other samples): en diverged 24: losses exp 5.6 obs 6, gains exp .6 obs 1 -> net exp -5.0 obs -5 (orig's lucky draw 84% regresses); zh net exp -1.4 obs -1. => control acc changes = regression to the mean, not an edit effect.
- Trigger reply-mean r (gen_1): en median -.010 (max -.001), zh -.003 (max +.034) vs ja-thinking +.87 -> L push in controls ~1% of ja.

## Pass 25 (gen_26 'both' = gen_11 composition; final confirmation) -> FINAL = G6+L (copy in /work/best_2)
- ja 60-99: orig think ja17/zh14/en9, G6+L ja38/zh2/en0 (paired +.525+-.08); ans-ja .275->.675 (+.40+-.10); acc .25->.25 (0+-.07). Same seeds G6 (gen_11) ja38, MF6 ja32.
- controls: en 12/12 en, acc 9->8; zh 12/12 zh, acc 5->7.
- Pooled held-out ja (gen_24+gen_26): G6+L think-ja 158/168 (94%), acc 45/168 (.268); G6 86/104 (83%), acc .269; orig 49/104 (47%), acc .269.
