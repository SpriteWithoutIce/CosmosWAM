#!/usr/bin/env python3
"""Test WandB connection and logging."""

import wandb
import time
import sys

print("Testing WandB connection...")

try:
    run = wandb.init(
        project="cosmos-wam-robotwin",
        name=f"test-{time.strftime('%Y%m%d-%H%M%S')}",
        settings=wandb.Settings(start_method="thread"),
    )
    
    print(f"WandB run created: {run.url}")
    
    # Log some test data
    for i in range(5):
        run.log({
            "test/metric": i * 10,
            "test/step": i,
        }, step=i)
        print(f"Logged step {i}")
        time.sleep(0.5)
    
    run.summary["test/final"] = 42
    run.finish()
    
    print("✓ WandB test successful!")
    print(f"Check: {run.url}")
    
except Exception as e:
    print(f"✗ WandB test failed: {e}")
    sys.exit(1)
