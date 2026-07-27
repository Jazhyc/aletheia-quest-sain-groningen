# Phoenix Wright 5.0 resolved-intent adapter

This is the rank-1, alpha-2 Qwen3.5-9B LoRA from
`results/blackbox/qwen9b_heterogeneous_resolved_intent_rank1_v1/adapter`.
Its `adapter_model.safetensors` SHA-256 is
`6380a77efdfc55b0fd0b011bb16bc4ca578272b137d05187de9460eca78d9679`.
Phoenix 5.0 applies its selected empty-summary binary-label margin only to
non-harm rows with a final-assistant reasoning trace. The frozen HP-KR
harm-pressure specialist route is exclusive and never invokes this adapter.
