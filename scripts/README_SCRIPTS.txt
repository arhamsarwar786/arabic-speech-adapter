========================================
ARABIC SPEECH ADAPTER - SCRIPT GUIDE
========================================

TWO OPTIONS:
1. Run WITHOUT Docker (direct on your machine)
2. Run WITH Docker (isolated environment)

========================================
OPTION 1: WITHOUT DOCKER
========================================

Step-by-step (numbered scripts):
  1. bash scripts/1_setup_environment.sh       # Setup Python 3.10 + venv + dependencies
  2. bash scripts/2_download_datasets.sh       # Download Arabic datasets (~20GB)
  3. bash scripts/3_preprocess_data.sh         # Preprocess to training format
  4. bash scripts/4_train_standard.sh          # Standard training
     OR
     bash scripts/5_train_high_gpu.sh          # Optimized for 40-100GB GPU

All-in-one:
  bash scripts/run_complete_pipeline.sh        # Runs all steps 1-4

Utilities:
  bash scripts/check_system_ready.sh           # Verify everything is ready
  bash scripts/test_without_data.sh            # Quick test without data download
  bash scripts/package_for_transfer.sh         # Create tarball for GPU machine
  bash scripts/setup_git_repo.sh               # Initialize git repository

========================================
OPTION 2: WITH DOCKER
========================================

Step-by-step (numbered scripts):
  1. bash scripts/docker_1_build_image.sh      # Build Docker image (10-15 min)
  2. bash scripts/docker_2_test_local.sh       # Test without GPU (verify setup)
  3. bash scripts/docker_3_run_training.sh     # Run training with GPU
  
Development:
  bash scripts/docker_4_interactive_shell.sh   # Open shell for debugging
  
Transfer to GPU machine:
  bash scripts/docker_5_save_image.sh          # Save image as file for transfer

========================================
WHICH OPTION TO CHOOSE?
========================================

WITHOUT Docker:
  ✅ Faster setup (5-10 min)
  ✅ Direct access to system
  ✅ Auto-installs Python 3.10 if missing
  ⚠️  May have environment differences

WITH Docker:
  ✅ Exact same environment everywhere
  ✅ Test locally before GPU machine
  ✅ No version conflicts
  ⏱️  Longer setup (10-15 min)

Recommendation:
  - Local testing: WITHOUT Docker (faster)
  - GPU machine: WITH Docker (reproducible)

========================================
QUICK START
========================================

Fastest way (WITHOUT Docker):
  bash scripts/1_setup_environment.sh && bash scripts/4_train_standard.sh

Guaranteed compatibility (WITH Docker):
  bash scripts/docker_1_build_image.sh && bash scripts/docker_3_run_training.sh

========================================
