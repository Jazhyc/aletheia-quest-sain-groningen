Phoenix Wright 4.x submission adapters

`resolved_intent/` contains the rank-1, alpha-2 LoRA selected for Phoenix
4.2's non-harm reasoning-intent margin. The ordinary Phoenix adapter remains under
`submission/phoenix_wright_v3_adapters/deception/`; Phoenix 4.4 replaces its
archived weights with the selected rank-16 GRPO continuation while retaining
the historical directory name for package compatibility.
