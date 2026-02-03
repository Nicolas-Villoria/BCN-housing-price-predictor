#!/usr/bin/env python3
"""
rollback_model.py
-----------------------
Model Rollback Script for Production Recovery

This script allows quick rollback to a previous model version when issues
are detected in production. It maintains a history of model versions and
can restore any previous version.

Features:
- List available model versions
- Rollback to a specific version
- Rollback to the previous version (N-1)
- Automatic backup of current model before rollback
- Validation of restored model

Usage:
    python scripts/rollback_model.py list              # List available versions
    python scripts/rollback_model.py rollback          # Rollback to previous version
    python scripts/rollback_model.py rollback v1.2.3   # Rollback to specific version
    python scripts/rollback_model.py backup            # Backup current model

Directory Structure:
    models/
    ├── champion_model.pkl          # Current production model
    ├── feature_transformer.pkl     # Current transformer
    ├── model_metadata.json         # Current metadata
    └── versions/                   # Version history
        ├── 20260127_181012/
        │   ├── champion_model.pkl
        │   ├── feature_transformer.pkl
        │   └── model_metadata.json
        └── 20260201_143022/
            └── ...
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
VERSIONS_DIR = MODELS_DIR / "versions"

# Model artifact files
MODEL_FILES = [
    "champion_model.pkl",
    "feature_transformer.pkl",
    "model_metadata.json"
]


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def ensure_versions_dir():
    """Ensure the versions directory exists."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_current_version() -> str:
    """Get the version string of the currently deployed model."""
    metadata_path = MODELS_DIR / "model_metadata.json"
    
    if not metadata_path.exists():
        return "unknown"
    
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata.get("version", "unknown")
    except Exception:
        return "unknown"


def list_versions() -> list:
    """
    List all available model versions.
    
    Returns:
        List of version info dicts sorted by date (newest first)
    """
    ensure_versions_dir()
    
    versions = []
    
    for version_dir in VERSIONS_DIR.iterdir():
        if not version_dir.is_dir():
            continue
        
        metadata_path = version_dir / "model_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                versions.append({
                    "version": metadata.get("version", version_dir.name),
                    "path": str(version_dir),
                    "model_type": metadata.get("model_type", "unknown"),
                    "training_date": metadata.get("training_date", "unknown"),
                    "metrics": metadata.get("metrics", {}),
                    "threshold_passed": metadata.get("threshold_passed", False)
                })
            except Exception as e:
                versions.append({
                    "version": version_dir.name,
                    "path": str(version_dir),
                    "error": str(e)
                })
        else:
            # Check if model files exist even without metadata
            if (version_dir / "champion_model.pkl").exists():
                versions.append({
                    "version": version_dir.name,
                    "path": str(version_dir),
                    "model_type": "unknown",
                    "training_date": "unknown"
                })
    
    # Sort by version (which is typically a timestamp)
    versions.sort(key=lambda x: x["version"], reverse=True)
    
    return versions


def backup_current_model() -> str|None:
    """
    Backup the current production model to the versions directory.
    
    Returns:
        Version string of the backed up model
    """
    ensure_versions_dir()
    
    current_version = get_current_version()
    
    if current_version == "unknown":
        # Generate a timestamp-based version
        current_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    backup_dir = VERSIONS_DIR / current_version
    
    # Check if backup already exists
    if backup_dir.exists():
        print(f"  Version {current_version} already backed up")
        return current_version
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all model files
    files_copied = 0
    for filename in MODEL_FILES:
        src = MODELS_DIR / filename
        dst = backup_dir / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            files_copied += 1
    
    if files_copied > 0:
        print(f" Backed up {files_copied} files to: {backup_dir}")
        return current_version
    else:
        # Remove empty backup directory
        backup_dir.rmdir()
        print(" No model files found to backup")
        return None


def restore_version(version: str) -> bool:
    """
    Restore a specific model version to production.
    
    Args:
        version: Version string to restore
        
    Returns:
        True if successful, False otherwise
    """
    version_dir = VERSIONS_DIR / version
    
    if not version_dir.exists():
        print(f" Version {version} not found")
        return False
    
    # First, backup current model
    print("\n Backing up current model...")
    backup_current_model()
    
    # Restore the specified version
    print(f"\n Restoring version {version}...")
    
    files_restored = 0
    for filename in MODEL_FILES:
        src = version_dir / filename
        dst = MODELS_DIR / filename
        
        if src.exists():
            shutil.copy2(src, dst)
            files_restored += 1
            print(f"    Restored {filename}")
        else:
            print(f"     {filename} not found in version")
    
    if files_restored == 0:
        print(" No files were restored")
        return False
    
    # Verify restoration
    new_version = get_current_version()
    print(f"\n Successfully restored to version: {new_version}")
    
    return True


def rollback_to_previous() -> bool:
    """
    Rollback to the previous model version (N-1).
    
    Returns:
        True if successful, False otherwise
    """
    versions = list_versions()
    current = get_current_version()
    
    if len(versions) < 1:
        print(" No previous versions available for rollback")
        return False
    
    # Find a version that's not the current one
    for v in versions:
        if v["version"] != current:
            return restore_version(v["version"])
    
    print(" No different version available for rollback")
    return False


# ==============================================================================
# CLI COMMANDS
# ==============================================================================

def cmd_list(args):
    """List all available model versions."""
    versions = list_versions()
    current = get_current_version()
    
    print("\n" + "="*70)
    print("📦 MODEL VERSION HISTORY")
    print("="*70)
    print(f"\n🎯 Current Production: {current}")
    print(f"\n{'Version':<20} {'Type':<15} {'R²':<8} {'RMSE':<12} {'Date'}")
    print("-"*70)
    
    if not versions:
        print("   No versions found in history")
    
    for v in versions:
        is_current = "→ " if v["version"] == current else "  "
        metrics = v.get("metrics", {})
        r2 = metrics.get("r2", "N/A")
        rmse = metrics.get("rmse", "N/A")
        
        r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
        rmse_str = f"€{rmse:,.0f}" if isinstance(rmse, (int, float)) else str(rmse)
        
        date = v.get("training_date", "N/A")
        if date != "N/A" and len(date) > 10:
            date = date[:10]  # Just the date part
        
        print(f"{is_current}{v['version']:<18} {v.get('model_type', 'N/A'):<15} "
              f"{r2_str:<8} {rmse_str:<12} {date}")
    
    print("="*70 + "\n")


def cmd_backup(args):
    """Backup current production model."""
    print("\n📦 BACKUP CURRENT MODEL")
    print("-"*40)
    
    version = backup_current_model()
    
    if version:
        print(f"\n✅ Backup complete: {version}")
        return 0
    return 1


def cmd_rollback(args):
    """Rollback to a previous version."""
    print("\n🔄 MODEL ROLLBACK")
    print("-"*40)
    
    current = get_current_version()
    print(f"Current version: {current}")
    
    if args.version:
        # Rollback to specific version
        target = args.version
        print(f"Target version: {target}")
        
        if target == current:
            print("⚠️  Target version is the same as current. No rollback needed.")
            return 0
        
        success = restore_version(target)
    else:
        # Rollback to previous version
        print("Rolling back to previous version...")
        success = rollback_to_previous()
    
    if success:
        print("\n" + "="*40)
        print("🎉 ROLLBACK SUCCESSFUL")
        print("="*40)
        print("\n⚠️  Remember to:")
        print("   1. Restart the API service")
        print("   2. Verify predictions are working")
        print("   3. Monitor for any issues")
        return 0
    else:
        print("\n❌ ROLLBACK FAILED")
        return 1


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model rollback utility for production recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/rollback_model.py list              # Show all versions
  python scripts/rollback_model.py backup            # Backup current model
  python scripts/rollback_model.py rollback          # Rollback to previous
  python scripts/rollback_model.py rollback 20260127_181012  # Rollback to specific
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available model versions")
    list_parser.set_defaults(func=cmd_list)
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup current model")
    backup_parser.set_defaults(func=cmd_backup)
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to previous version")
    rollback_parser.add_argument(
        "version",
        nargs="?",
        help="Specific version to rollback to (optional, defaults to previous)"
    )
    rollback_parser.set_defaults(func=cmd_rollback)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
