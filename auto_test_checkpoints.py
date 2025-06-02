#!/usr/bin/env python3
"""
Automatic checkpoint testing for FLUX LoRA training
Tests each saved checkpoint with your custom prompt
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime


def test_checkpoint(checkpoint_path, prompt, output_dir):
    """Test a specific checkpoint"""
    
    checkpoint_name = Path(checkpoint_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/test_{checkpoint_name}_{timestamp}.png"
    
    print(f"🎨 Testing checkpoint: {checkpoint_name}")
    print(f"📝 Prompt: {prompt}")
    
    # Test command
    cmd = [
        "python", "inference.py",
        "--model", "flux",
        "--model_path", checkpoint_path,
        "--prompt", prompt,
        "--output", output_file,
        "--height", "1024",
        "--width", "1024",
        "--guidance_scale", "7.5",
        "--num_steps", "20"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ Test successful: {output_file}")
            return output_file
        else:
            print(f"❌ Test failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Test timed out for {checkpoint_name}")
        return None
    except Exception as e:
        print(f"❌ Error testing {checkpoint_name}: {e}")
        return None


def monitor_training(output_dir, prompt, test_interval=500):
    """Monitor training and test checkpoints automatically"""
    
    print("🔍 Starting automatic checkpoint monitoring...")
    print(f"📁 Monitoring: {output_dir}")
    print(f"📝 Test prompt: {prompt}")
    print(f"⏱️  Test interval: every {test_interval} steps")
    print("=" * 60)
    
    # Create test output directory
    test_output_dir = f"{output_dir}/test_images"
    Path(test_output_dir).mkdir(parents=True, exist_ok=True)
    
    tested_checkpoints = set()
    
    while True:
        try:
            # Look for new checkpoints
            if os.path.exists(output_dir):
                checkpoint_pattern = f"{output_dir}/checkpoint-*"
                checkpoints = list(Path(output_dir).glob("checkpoint-*"))
                
                # Sort by step number
                checkpoints.sort(key=lambda x: int(x.name.split('-')[1]) if x.name.split('-')[1].isdigit() else 0)
                
                for checkpoint in checkpoints:
                    if checkpoint not in tested_checkpoints:
                        # Test this checkpoint
                        result = test_checkpoint(str(checkpoint), prompt, test_output_dir)
                        
                        if result:
                            tested_checkpoints.add(checkpoint)
                            print(f"📊 Progress: {len(tested_checkpoints)} checkpoints tested")
                        
                        time.sleep(5)  # Brief pause between tests
            
            # Check if training is complete
            final_model_path = f"{output_dir}/final_model"
            if os.path.exists(final_model_path):
                print(f"\n🎉 Training complete! Testing final model...")
                final_result = test_checkpoint(final_model_path, prompt, test_output_dir)
                if final_result:
                    print(f"✅ Final model test: {final_result}")
                break
            
            # Wait before next check
            time.sleep(30)
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in monitoring: {e}")
            time.sleep(60)
    
    print(f"\n📊 Summary:")
    print(f"  Tested checkpoints: {len(tested_checkpoints)}")
    print(f"  Test images saved to: {test_output_dir}")


def main():
    """Main function"""
    
    # Default settings
    default_output_dir = "/workspace/outputs/flux_qlora"
    default_prompt = "a solo indian girl wearing a tshirt is standing in a room"
    default_interval = 500
    
    print("🎨 FLUX LoRA Checkpoint Auto-Tester")
    print("=" * 50)
    
    # Get user inputs
    output_dir = input(f"Training output directory [{default_output_dir}]: ").strip() or default_output_dir
    prompt = input(f"Test prompt [{default_prompt}]: ").strip() or default_prompt
    interval = input(f"Test interval (steps) [{default_interval}]: ").strip()
    interval = int(interval) if interval.isdigit() else default_interval
    
    # Validate output directory
    if not os.path.exists(output_dir):
        print(f"⚠️  Output directory doesn't exist yet: {output_dir}")
        print(f"Will start monitoring when training begins...")
    
    # Start monitoring
    monitor_training(output_dir, prompt, interval)


if __name__ == "__main__":
    main()
