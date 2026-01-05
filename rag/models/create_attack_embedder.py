#!/usr/bin/env python3
"""
Create Attack-specific embedder for ML Defender.

Input:  83 network traffic features (float32)
Output: 256-d attack embedding (float32)

Architecture: Focused on attack pattern detection
Note: Smaller dimension for class-separated indices strategy

Via Appia Quality: Synthetic model with correct architecture
                   to validate pipeline, not for production.
"""

import torch
import torch.nn as nn
import onnx

class AttackEmbedder(nn.Module):
    """
    Attack embedder: 83 features → 256-d
    
    Specialized for attack pattern detection
    Smaller dimension for class-separated indices
    Uses BatchNorm for faster inference
    """
    def __init__(self, input_dim=83, hidden_dim=128, output_dim=256):
        super().__init__()
        
        self.network = nn.Sequential(
            # First layer: Feature extraction
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            # Second layer: Pattern detection
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            
            # Output layer: Attack embedding
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
    
    def forward(self, x):
        return self.network(x)

def main():
    print("╔════════════════════════════════════════╗")
    print("║  Creating Attack Embedder (83→256-d)  ║")
    print("╚════════════════════════════════════════╝\n")
    
    print("Step 1: Initializing Attack architecture...")
    model = AttackEmbedder(input_dim=83, output_dim=256)
    model.eval()
    print("  ✅ Model initialized (83 → 256-d)\n")
    
    print("Step 2: Creating export input...")
    dummy_input = torch.randn(1, 83)
    print(f"  ✅ Input shape: {dummy_input.shape}\n")
    
    print("Step 3: Exporting to ONNX...")
    torch.onnx.export(
        model, dummy_input, "attack_embedder.onnx",
        input_names=['features'],
        output_names=['embedding'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        },
        opset_version=18,
        verbose=False
    )
    print("  ✅ Exported: attack_embedder.onnx\n")
    
    print("Step 4: Verifying model...")
    onnx_model = onnx.load("attack_embedder.onnx")
    onnx.checker.check_model(onnx_model)
    print("  ✅ Model verified\n")
    
    print("Model Information:")
    print("  Input:  features (batch, 83)")
    print("  Output: embedding (batch, 256)")
    print("  Type:   Attack-specific embedder")
    print("  Note:   Smaller dim for separated indices strategy")
    print("  Status: Synthetic model for pipeline validation")
    print("\n╔════════════════════════════════════════╗")
    print("║  Attack Embedder Created ✅            ║")
    print("╚════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
