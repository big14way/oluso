# Oluso

**Measuring and repairing safety and calibration drift in Nigerian languages, without retraining.**

Oluso (Olùṣọ́, guardian) evaluates a small open-weight multilingual model on Yoruba, Igbo, Hausa and Nigerian Pidgin, including code-mixed prompts, on two failure modes at once: safety drift (refusal collapse and over-refusal across languages) and calibration drift (confidence that no longer tracks accuracy). It then repairs both at inference time with activation steering and per-language temperature scaling, and reports before and after results on four metrics.

## Status

In active development. Setup instructions, architecture and a demo will land here as the build progresses.

## License

MIT
