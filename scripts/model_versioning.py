"""
model_versioning.py
-----------------------
Enhanced Model Versioning for MLOps

This module provides comprehensive model versioning with:
- Semantic versioning (major.minor.patch)
- Git commit tracking
- Data lineage (training data hash)
- Environment fingerprinting
- Deployment history
- Model lineage (parent version tracking)

Usage:
    from model_versioning import ModelVersionManager
    
    manager = ModelVersionManager()
    metadata = manager.create_version_metadata(
        model=trained_model,
        metrics={"rmse": 100000, "r2": 0.90},
        model_type="RandomForest"
    )
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
VERSIONS_DIR = MODELS_DIR / "versions"
VERSION_REGISTRY_PATH = MODELS_DIR / "version_registry.json"


# ==============================================================================
# VERSION MANAGER CLASS
# ==============================================================================

class ModelVersionManager:
    """
    Manages model versioning with comprehensive metadata tracking.
    
    Features:
    - Semantic versioning with auto-increment
    - Git integration for commit tracking
    - Data fingerprinting
    - Environment capture
    - Deployment history
    """
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.versions_dir = models_dir / "versions"
        self.registry_path = models_dir / "version_registry.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._load_registry()
    
    def _load_registry(self):
        """Load or initialize the version registry."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "current_version": "0.0.0",
                "versions": [],
                "deployments": []
            }
    
    def _save_registry(self):
        """Persist the version registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_git_info(self) -> dict:
        """
        Get current git repository information.
        
        Returns:
            Dict with commit hash, branch, author, and dirty status
        """
        git_info = {
            "commit_hash": None,
            "commit_short": None,
            "branch": None,
            "author": None,
            "commit_date": None,
            "is_dirty": None,
            "remote_url": None
        }
        
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode == 0:
                git_info["commit_hash"] = result.stdout.strip()
                git_info["commit_short"] = result.stdout.strip()[:8]
            
            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()
            
            # Get author
            result = subprocess.run(
                ["git", "log", "-1", "--format=%an"],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode == 0:
                git_info["author"] = result.stdout.strip()
            
            # Get commit date
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ci"],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode == 0:
                git_info["commit_date"] = result.stdout.strip()
            
            # Check if working directory is dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            git_info["is_dirty"] = len(result.stdout.strip()) > 0
            
            # Get remote URL
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode == 0:
                git_info["remote_url"] = result.stdout.strip()
                
        except Exception as e:
            git_info["error"] = str(e)
        
        return git_info
    
    def get_environment_info(self) -> dict:
        """
        Capture current environment information.
        
        Returns:
            Dict with Python version, OS, and key package versions
        """
        env_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": platform.node(),
            "packages": {}
        }
        
        # Get key ML package versions
        packages_to_check = [
            "scikit-learn", "pandas", "numpy", "joblib",
            "fastapi", "uvicorn", "pydantic"
        ]
        
        for pkg in packages_to_check:
            try:
                import importlib.metadata
                env_info["packages"][pkg] = importlib.metadata.version(pkg)
            except Exception:
                pass
        
        return env_info
    
    def compute_data_hash(self, data_path: Path) -> Optional[str]:
        """
        Compute a hash of the training data for lineage tracking.
        
        Args:
            data_path: Path to data directory or file
            
        Returns:
            SHA256 hash of the data
        """
        try:
            hasher = hashlib.sha256()
            
            if data_path.is_file():
                with open(data_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        hasher.update(chunk)
            elif data_path.is_dir():
                # Hash all CSV files in directory
                for csv_file in sorted(data_path.rglob("*.csv")):
                    with open(csv_file, 'rb') as f:
                        for chunk in iter(lambda: f.read(8192), b''):
                            hasher.update(chunk)
            
            return hasher.hexdigest()[:16]  # First 16 chars for brevity
            
        except Exception as e:
            return None
    
    def get_next_version(self, bump: str = "patch") -> str:
        """
        Calculate the next semantic version.
        
        Args:
            bump: One of "major", "minor", "patch"
            
        Returns:
            Next version string (e.g., "1.2.4")
        """
        current = self.registry.get("current_version", "0.0.0")
        parts = [int(x) for x in current.split(".")]
        
        if len(parts) != 3:
            parts = [0, 0, 0]
        
        major, minor, patch = parts
        
        if bump == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        return f"{major}.{minor}.{patch}"
    
    def create_version_metadata(
        self,
        model_type: str,
        metrics: dict,
        training_samples: int = 0,
        data_path: Optional[Path] = None,
        parent_version: Optional[str] = None,
        bump: str = "patch",
        description: str = "",
        tags: list = None
    ) -> dict:
        """
        Create comprehensive version metadata.
        
        Args:
            model_type: Type of model (e.g., "RandomForest")
            metrics: Performance metrics dict
            training_samples: Number of training samples
            data_path: Path to training data (for hashing)
            parent_version: Previous version this was trained from
            bump: Version bump type ("major", "minor", "patch")
            description: Human-readable description
            tags: List of tags (e.g., ["production", "stable"])
            
        Returns:
            Complete metadata dict
        """
        new_version = self.get_next_version(bump)
        timestamp = datetime.now()
        
        metadata = {
            # Version identification
            "version": new_version,
            "version_id": timestamp.strftime("%Y%m%d_%H%M%S"),
            "semantic_version": new_version,
            
            # Model information
            "model_type": model_type,
            "description": description,
            "tags": tags or [],
            
            # Performance metrics
            "metrics": {
                "rmse": round(metrics.get("rmse", 0), 2),
                "r2": round(metrics.get("r2", 0), 4),
                "mae": round(metrics.get("mae", 0), 2),
                "mape": round(metrics.get("mape", 0), 2) if "mape" in metrics else None
            },
            
            # Training information
            "training": {
                "date": timestamp.isoformat(),
                "samples": training_samples,
                "data_hash": self.compute_data_hash(data_path) if data_path else None
            },
            
            # Lineage
            "lineage": {
                "parent_version": parent_version,
                "created_from": "train_sklearn.py"
            },
            
            # Git information
            "git": self.get_git_info(),
            
            # Environment
            "environment": self.get_environment_info(),
            
            # Thresholds
            "validation": {
                "rmse_threshold": 150000,
                "r2_threshold": 0.70,
                "threshold_passed": (
                    metrics.get("rmse", float("inf")) <= 150000 and 
                    metrics.get("r2", 0) >= 0.70
                )
            }
        }
        
        # Update registry
        self.registry["current_version"] = new_version
        self.registry["versions"].append({
            "version": new_version,
            "version_id": metadata["version_id"],
            "created_at": timestamp.isoformat(),
            "model_type": model_type,
            "metrics": metadata["metrics"],
            "git_commit": metadata["git"].get("commit_short")
        })
        
        self._save_registry()
        
        return metadata
    
    def record_deployment(
        self,
        version: str,
        environment: str = "production",
        deployed_by: str = None,
        notes: str = ""
    ):
        """
        Record a deployment event.
        
        Args:
            version: Version being deployed
            environment: Deployment environment
            deployed_by: Who deployed (or "ci/cd")
            notes: Deployment notes
        """
        deployment = {
            "version": version,
            "environment": environment,
            "deployed_at": datetime.now().isoformat(),
            "deployed_by": deployed_by or os.getenv("USER", "unknown"),
            "notes": notes
        }
        
        self.registry["deployments"].append(deployment)
        self._save_registry()
        
        print(f" Recorded deployment: v{version} → {environment}")
    
    def get_version_history(self, limit: int = 10) -> list:
        """Get recent version history."""
        return self.registry.get("versions", [])[-limit:]
    
    def get_deployment_history(self, limit: int = 10) -> list:
        """Get recent deployment history."""
        return self.registry.get("deployments", [])[-limit:]
    
    def get_current_version(self) -> str:
        """Get the current semantic version."""
        return self.registry.get("current_version", "0.0.0")


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def enhance_existing_metadata(metadata_path: Path) -> dict:
    """
    Enhance an existing model_metadata.json with versioning info.
    
    Args:
        metadata_path: Path to existing metadata file
        
    Returns:
        Enhanced metadata dict
    """
    manager = ModelVersionManager()
    
    # Load existing metadata
    with open(metadata_path) as f:
        existing = json.load(f)
    
    # Add versioning information
    enhanced = {
        **existing,
        "semantic_version": manager.get_current_version(),
        "git": manager.get_git_info(),
        "environment": manager.get_environment_info(),
        "enhanced_at": datetime.now().isoformat()
    }
    
    return enhanced


def print_version_info():
    """Print current version information."""
    manager = ModelVersionManager()
    
    print("\n" + "="*60)
    print("📦 MODEL VERSION INFORMATION")
    print("="*60)
    
    print(f"\nCurrent Version: v{manager.get_current_version()}")
    
    git = manager.get_git_info()
    if git.get("commit_short"):
        print(f"Git Commit: {git['commit_short']} ({git.get('branch', 'unknown')})")
        print(f"Author: {git.get('author', 'unknown')}")
        if git.get("is_dirty"):
            print("⚠️  Working directory has uncommitted changes")
    
    print("\nRecent Versions:")
    for v in manager.get_version_history(5):
        print(f"  v{v['version']} - {v['model_type']} - R²={v['metrics'].get('r2', 'N/A')}")
    
    print("\nRecent Deployments:")
    for d in manager.get_deployment_history(5):
        print(f"  v{d['version']} → {d['environment']} ({d['deployed_at'][:10]})")
    
    print("="*60)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model versioning utilities")
    parser.add_argument("command", choices=["info", "bump", "deploy"], 
                       help="Command to run")
    parser.add_argument("--type", choices=["major", "minor", "patch"], 
                       default="patch", help="Version bump type")
    parser.add_argument("--env", default="production", 
                       help="Deployment environment")
    
    args = parser.parse_args()
    
    if args.command == "info":
        print_version_info()
    elif args.command == "bump":
        manager = ModelVersionManager()
        new_version = manager.get_next_version(args.type)
        print(f"Next {args.type} version would be: v{new_version}")
    elif args.command == "deploy":
        manager = ModelVersionManager()
        current = manager.get_current_version()
        manager.record_deployment(current, args.env)
