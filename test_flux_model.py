#!/usr/bin/env python3
"""
Test FLUX model with your specific prompt
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def test_flux_model(model_path=None, prompt=None):
    """Test FLUX model with custom prompt"""
    
    # Default values
    default_prompt = "a solo indian girl wearing a tshirt is standing in a room"
    default_model_path = "/workspace/outputs/flux_qlora"
    
    if prompt is None:
        prompt = default_prompt
    
    if model_path is None:
        model_path = default_model_path
    
    print("🎨 FLUX Model Tester")
    print("=" * 50)
    print(f"📝 Prompt: {prompt}")
    print(f"🤖 Model: {model_path}")
    
    # Check if model exists
    model_path = Path(model_path)
    
    # Look for different model formats
    possible_paths = [
        model_path / "final_model",
        model_path / "lora",
        model_path,
    ]
    
    # Also check for checkpoints
    if model_path.exists():
        checkpoints = list(model_path.glob("checkpoint-*"))
        if checkpoints:
            # Use latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]) if x.name.split('-')[1].isdigit() else 0)
            possible_paths.insert(0, latest_checkpoint)
    
    # Find valid model path
    valid_model_path = None
    for path in possible_paths:
        if path.exists():
            valid_model_path = path
            break
    
    if not valid_model_path:
        print(f"❌ No model found in: {model_path}")
        print(f"Available paths checked:")
        for path in possible_paths:
            print(f"  - {path} {'✅' if path.exists() else '❌'}")
        return
    
    print(f"✅ Using model: {valid_model_path}")
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/workspace/outputs/test_generation_{timestamp}.png"
    
    # Build inference command
    cmd = [
        "python", "inference.py",
        "--model", "flux",
        "--model_path", str(valid_model_path),
        "--prompt", prompt,
        "--output", output_file,
        "--height", "1024",
        "--width", "1024",
        "--guidance_scale", "7.5",
        "--num_steps", "20"
    ]
    
    print(f"\n🚀 Generating image...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ Generation successful!")
            print(f"📁 Image saved to: {output_file}")
            
            # Check if file actually exists
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / 1024  # KB
                print(f"📊 File size: {file_size:.1f} KB")
            else:
                print(f"⚠️  Output file not found: {output_file}")
                
        else:
            print(f"❌ Generation failed!")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Generation timed out (>5 minutes)")
    except Exception as e:
        print(f"❌ Error during generation: {e}")


def test_all_checkpoints():
    """Test all available checkpoints"""
    
    output_dir = Path("/workspace/outputs/flux_qlora")
    prompt = "a solo indian girl wearing a tshirt is standing in a room"
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    # Find all checkpoints
    checkpoints = list(output_dir.glob("checkpoint-*"))
    checkpoints.sort(key=lambda x: int(x.name.split('-')[1]) if x.name.split('-')[1].isdigit() else 0)
    
    # Check for final model
    final_model = output_dir / "final_model"
    if final_model.exists():
        checkpoints.append(final_model)
    
    if not checkpoints:
        print(f"❌ No checkpoints found in: {output_dir}")
        return
    
    print(f"🔍 Found {len(checkpoints)} checkpoints to test")
    
    for i, checkpoint in enumerate(checkpoints):
        print(f"\n{'='*60}")
        print(f"Testing {i+1}/{len(checkpoints)}: {checkpoint.name}")
        print(f"{'='*60}")
        
        test_flux_model(str(checkpoint), prompt)
        
        if i < len(checkpoints) - 1:
            input("Press Enter to continue to next checkpoint...")


def main():
    """Main function"""
    
    print("🎨 FLUX Model Testing Options")
    print("=" * 50)
    print("1. Test latest model/checkpoint")
    print("2. Test specific model path")
    print("3. Test all checkpoints")
    print("4. Custom prompt test")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        test_flux_model()
        
    elif choice == "2":
        model_path = input("Enter model path: ").strip()
        test_flux_model(model_path)
        
    elif choice == "3":
        test_all_checkpoints()
        
    elif choice == "4":
        prompt = input("Enter custom prompt: ").strip()
        test_flux_model(prompt=prompt)
        
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
