#!/bin/bash
# Workshop paper review script — 0420-AM checkpoint
# This paper (VAE-Induced Semantic Drift in Video Generation: Disease, Diagnostic, and Treatment)
# is the consolidated CNLSA + TrACE-Video workshop submission.
# It requires: pip install torch torchvision timm transformers open_clip scikit-learn

echo "=== CNLSA / TrACE-Video Workshop Paper v3 (0420-AM) ==="
echo "Paper: workshop-paper-v3.md"
echo ""
echo "Key corrections from 0420-AM Scalpel review:"
echo "  - Abstract/Results: r=0.5472 -> r=0.3681 (verified in cnlsa_bridge_results.json)"
echo "  - CI: [0.36,0.74] -> [0.27,0.46]"
echo "  - R²: 0.37 -> 0.14"
echo "  - Section 3.3 framing: 'CLIP-specific' -> 'architecture-variant'"
echo "  - VAE model description fixed to match actual code"
echo ""
echo "New papers added to Related Work (Feb-Apr 2026):"
echo "  - LSA (2602.05966), Re2Pix (2604.11707), Diagonal Distillation (2603.09488)"
echo "  - Event-Driven Video (2603.13402), VGGRPO (2603.26599), EvoSearch (OpenReview)"
echo "  - LongLive (ICLR 2026), VideoGPA (2601.2328), FreeViS (2510.01686)"
echo "  - 'When Backdoors Go Beyond Triggers' (2602.20193)"
echo ""
echo "No executable code — paper review only."
echo "To view the paper: cat workshop-paper-v3.md"
