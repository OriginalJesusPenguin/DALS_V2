<!-- 779e9df1-bf93-4d8b-98b7-d74a2c39e4ab d1fb7eec-c7e6-4463-9d86-853b73706153 -->
# Add Kendall-Gal Loss Weighting

1. parser-flag: Extend `MeshDecoderTrainer.add_argparse_args` to include a boolean `--use_uncertainty_weights` flag (default False) for enabling Kendall & Gal style weighting.
2. init-params: In `MeshDecoderTrainer.__init__`, capture the flag, initialize learnable log-sigma parameters for chamfer, edge, quality, and norm losses when enabled, and register them with the optimizer.
3. loss-compute: Update the training loss assembly so that when the flag is true the four targeted losses use the Kendall-Gal formulation `L_i/(2σ_i^2)+log σ_i`, while the traditional weighted sum is kept for the other losses (or when the flag is false).
4. logging-adjust: Ensure logging/progress summaries reflect whichever weighting mode is active, including reporting sigma values when uncertainty weighting is enabled.