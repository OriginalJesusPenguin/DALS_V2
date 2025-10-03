#!/bin/bash
"""
Example script showing how to use both UMAP plotting modes.
"""

echo "=== UMAP Visualization Examples ==="
echo ""

echo "1. Severity Mode (default) - Shows cirrhosis severity levels:"
echo "   - 0: Healthy (blue)"
echo "   - 1: Mild Cirrhosis (yellow)" 
echo "   - 2: Moderate Cirrhosis (orange)"
echo "   - 3: Severe Cirrhosis (red)"
echo ""
python plot_latent_umap.py \
    --latent_file latent_vectors.pt \
    --output umap_severity_example.png \
    --plot_mode severity \
    --n_neighbors 15 \
    --min_dist 0.1

echo ""
echo "2. Binary Mode - Shows cirrhotic vs healthy:"
echo "   - Healthy samples (blue)"
echo "   - Cirrhotic samples (red)"
echo ""
python plot_latent_umap.py \
    --latent_file latent_vectors.pt \
    --output umap_binary_example.png \
    --plot_mode binary \
    --n_neighbors 15 \
    --min_dist 0.1

echo ""
echo "=== Examples Complete ==="
echo "Generated files:"
echo "  - umap_severity_example.png (severity-based plot)"
echo "  - umap_severity_example_combined_density.png"
echo "  - umap_binary_example.png (binary plot)"
echo "  - umap_binary_example_combined_density.png"
