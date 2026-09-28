# e4b-a1b-bench

Independent multiple-choice benchmark of **Gemma 4 E4B** builds on the same question set, English and
Russian: 570 questions from Cohere's Global-MMLU (10 per MMLU subject, fixed seed) and the SAME 570 in
Russian (1 140 prompts, `data/bench_gmmlu.jsonl`). No-think chat, one-letter answers, greedy.

`.github/workflows/gguf-mc.yml` runs any GGUF of the model through a CPU llama.cpp server in 8 parallel
shards and uploads each shard's replies as an artifact. It is one of several harnesses answering the
same prompts (the bf16 reference and sparsified-FFN variants run elsewhere).

Data: Global-MMLU (CohereLabs, Apache-2.0), derived from MMLU (Hendrycks et al.).
