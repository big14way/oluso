Oluso measures and repairs, without retraining, two kinds of drift in Tiny Aya Earth across Yoruba, Igbo, Hausa, Nigerian Pidgin and Pidgin-English code-mix: safety drift (lost refusals, or refused harmless requests) and calibration drift (confidence that stops tracking accuracy).

We locked a test set of 1,600 prompts: TukaBench for English, Yoruba, Igbo and Hausa, plus new Pidgin and code-mix sets. Each answer is labelled refused, complied or deflected. The judge is WildGuard reading NLLB/MAFAND translations, validated on English human labels (κ 0.88 XSTest, 0.77 JailbreakBench) and cross-checked by a native-language refusal lexicon. Calibration is measured on AfriMMLU with ECE, Brier, NLL and AUROC.

Before: the model gives harmful content in answer to 3% of harmful English requests, but 35% in Yoruba, 24% in Igbo and 15% in Hausa, and 58% on culturally grounded Yoruba requests. Calibration drifts differently: the model is overconfident in every language, English included (ECE 0.19–0.23).

Repair: conditional activation steering (CAST) with SafeTune. A refusal direction is added only when a gate on the model's own activations flags the prompt. Layer 18 and α = 1 were chosen on a dev split by a rule committed before any steered output existed. The test set was run once, with a random-direction control and ungated addition. Calibration is repaired with one temperature fitted by likelihood on held-out items.

After, pooled over the non-English settings with paired, clustered statistics:

- Harmful answers fall 6 points [3, 9], and on culturally grounded requests 17 points [12, 22]. Hausa falls from 15% to 4% (Holm p = 0.006).
- A random direction with the same gate and norm does nothing; CAST beats it by 7 points (p < 0.001).
- Over-refusal rises 13 points [10, 16], failing our 5-point non-inferiority margin. The gate, fitted against OR-Bench prompts, fires on almost every benign test prompt in Yoruba and Igbo, whose benign items are near-twins of the harmful ones. Where the gate is selective (English, Pidgin, code-mix) it cuts the over-refusal rise of ungated steering by more than half.
- One temperature (T = 2) brings ECE to its noise floor in Yoruba, Igbo and Hausa (0.05). The gate never fires on the quiz prompts, so accuracy and calibration are untouched; ungated steering costs up to 4 points of accuracy.

The repair ships as a guard under 20 KB (one vector, one gate, one temperature) for an unchanged model. It costs 12% of throughput, about $135 per million requests on one A10G: on par with gpt-4o-mini and 1.8 times the cheapest closed API. Code, data, the selection record and per-item results are public. The next step is to fit the gate on harmful/benign near-twins in each language.
