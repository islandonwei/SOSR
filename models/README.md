# Model artifacts

The HSM base encoder is sentence-transformers/paraphrase-multilingual-mpnet-base-v2. Training can download it through Hugging Face or accept a local --model directory.

Large weights are external. checkpoints.json records hashes and selected epochs for the original HSM and ablations. Training creates checkpoints/stage2_best.pt; evaluation loads it explicitly.

Qwen uses Qwen/Qwen2.5-7B-Instruct. Qwen-City requires the checkpoint from the included CityInstruction trainer or the authors' separately shared artifact. Its plan template records base-shard hashes and the inspected model revision.

Distribute checkpoints through release/model-hosting assets with their hashes and the base tokenizer/configuration.
