#!/usr/bin/env python3
"""
s3_storage.py
-----------------------
AWS S3 Storage for Model Artifacts

This module provides S3 integration for model artifact storage, enabling:
- Upload models to S3 after training
- Download models from S3 for deployment
- List available model versions in S3
- Sync local and remote model storage

Environment Variables Required:
    AWS_ACCESS_KEY_ID     : Your AWS access key
    AWS_SECRET_ACCESS_KEY : Your AWS secret key
    AWS_REGION            : AWS region (default: eu-west-1)
    S3_BUCKET_NAME        : S3 bucket name (default: bcn-rental-models)

Usage:
    # Upload current model
    python scripts/s3_storage.py upload
    
    # Download specific version
    python scripts/s3_storage.py download --version 1.0.0
    
    # List all versions in S3
    python scripts/s3_storage.py list
    
    # Sync (upload if local is newer)
    python scripts/s3_storage.py sync
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# S3 Configuration (can be overridden by environment variables)
DEFAULT_BUCKET_NAME = "bcn-rental-models"
DEFAULT_REGION = "eu-west-1"
S3_PREFIX = "models/"  # Folder prefix in S3

# Model files to sync
MODEL_FILES = [
    "champion_model.pkl",
    "feature_transformer.pkl",
    "model_metadata.json"
]


# ==============================================================================
# S3 CLIENT CLASS
# ==============================================================================

class S3ModelStorage:
    """
    Manages model artifact storage in AWS S3.
    
    Supports:
    - Uploading model artifacts with version tagging
    - Downloading specific versions
    - Listing available versions
    - Version comparison for sync
    """
    
    def __init__(
        self,
        bucket_name: str = None,
        region: str = None,
        models_dir: Path = MODELS_DIR
    ):
        """
        Initialize S3 storage client.
        
        Args:
            bucket_name: S3 bucket name (or env var S3_BUCKET_NAME)
            region: AWS region (or env var AWS_REGION)
            models_dir: Local models directory
        """
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME", DEFAULT_BUCKET_NAME)
        self.region = region or os.getenv("AWS_REGION", DEFAULT_REGION)
        self.models_dir = models_dir
        self.s3_client = None
        
        self._init_client()
    
    def _init_client(self):
        """Initialize the boto3 S3 client."""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError
            
            self.s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✅ Connected to S3 bucket: {self.bucket_name}")
            
        except ImportError:
            print("❌ boto3 not installed. Run: pip install boto3")
            self.s3_client = None
        except Exception as e:
            print(f"⚠️  S3 connection warning: {e}")
            # Don't fail - allow offline operation
    
    def is_connected(self) -> bool:
        """Check if S3 client is properly connected."""
        return self.s3_client is not None
    
    def _get_local_version(self) -> Optional[str]:
        """Get the version of the local model."""
        metadata_path = self.models_dir / "model_metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata.get("version") or metadata.get("semantic_version")
        except Exception:
            return None
    
    def upload_model(
        self,
        version: str = None,
        force: bool = False
    ) -> bool:
        """
        Upload model artifacts to S3.
        
        Args:
            version: Version tag (defaults to local model version)
            force: Overwrite existing version in S3
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            print("❌ S3 client not connected")
            return False
        
        # Get version from local metadata if not provided
        version = version or self._get_local_version()
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"⚠️  No version found, using timestamp: {version}")
        
        s3_version_prefix = f"{S3_PREFIX}{version}/"
        
        # Check if version already exists
        if not force:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=s3_version_prefix,
                    MaxKeys=1
                )
                if response.get("KeyCount", 0) > 0:
                    print(f"⚠️  Version {version} already exists in S3. Use --force to overwrite.")
                    return False
            except Exception:
                pass  # Continue if check fails
        
        print(f"\n📤 Uploading model v{version} to S3...")
        print(f"   Bucket: {self.bucket_name}")
        print(f"   Prefix: {s3_version_prefix}")
        
        uploaded = 0
        for filename in MODEL_FILES:
            local_path = self.models_dir / filename
            
            if not local_path.exists():
                print(f"   ⚠️  {filename} not found, skipping")
                continue
            
            s3_key = f"{s3_version_prefix}{filename}"
            
            try:
                self.s3_client.upload_file(
                    str(local_path),
                    self.bucket_name,
                    s3_key,
                    ExtraArgs={
                        "Metadata": {
                            "version": version,
                            "uploaded_at": datetime.now().isoformat(),
                            "uploaded_by": os.getenv("USER", "unknown")
                        }
                    }
                )
                print(f"   ✅ Uploaded {filename}")
                uploaded += 1
            except Exception as e:
                print(f"   ❌ Failed to upload {filename}: {e}")
        
        # Also upload to "latest" folder
        if uploaded > 0:
            print(f"\n📤 Updating 'latest' pointer...")
            for filename in MODEL_FILES:
                local_path = self.models_dir / filename
                if local_path.exists():
                    try:
                        self.s3_client.upload_file(
                            str(local_path),
                            self.bucket_name,
                            f"{S3_PREFIX}latest/{filename}"
                        )
                    except Exception:
                        pass
            
            print(f"\n✅ Successfully uploaded {uploaded} files to S3")
            return True
        
        return False
    
    def download_model(
        self,
        version: str = "latest",
        output_dir: Path = None
    ) -> bool:
        """
        Download model artifacts from S3.
        
        Args:
            version: Version to download (or "latest")
            output_dir: Directory to save files (defaults to models_dir)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            print("❌ S3 client not connected")
            return False
        
        output_dir = output_dir or self.models_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        s3_version_prefix = f"{S3_PREFIX}{version}/"
        
        print(f"\n📥 Downloading model v{version} from S3...")
        print(f"   Bucket: {self.bucket_name}")
        print(f"   Prefix: {s3_version_prefix}")
        
        downloaded = 0
        for filename in MODEL_FILES:
            s3_key = f"{s3_version_prefix}{filename}"
            local_path = output_dir / filename
            
            try:
                self.s3_client.download_file(
                    self.bucket_name,
                    s3_key,
                    str(local_path)
                )
                print(f"   ✅ Downloaded {filename}")
                downloaded += 1
            except Exception as e:
                print(f"   ❌ Failed to download {filename}: {e}")
        
        if downloaded > 0:
            print(f"\n✅ Successfully downloaded {downloaded} files from S3")
            return True
        
        return False
    
    def list_versions(self) -> List[dict]:
        """
        List all model versions available in S3.
        
        Returns:
            List of version info dicts
        """
        if not self.is_connected():
            print("❌ S3 client not connected")
            return []
        
        versions = []
        
        try:
            # List all "folders" under the models prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=S3_PREFIX,
                Delimiter='/'
            ):
                for prefix in page.get('CommonPrefixes', []):
                    version_prefix = prefix['Prefix']
                    version_name = version_prefix.replace(S3_PREFIX, '').rstrip('/')
                    
                    if version_name == 'latest':
                        continue
                    
                    # Try to get metadata
                    try:
                        metadata_response = self.s3_client.get_object(
                            Bucket=self.bucket_name,
                            Key=f"{version_prefix}model_metadata.json"
                        )
                        metadata = json.loads(metadata_response['Body'].read().decode('utf-8'))
                        
                        versions.append({
                            "version": version_name,
                            "s3_prefix": version_prefix,
                            "model_type": metadata.get("model_type"),
                            "metrics": metadata.get("metrics", {}),
                            "training_date": metadata.get("training_date") or metadata.get("training", {}).get("date"),
                            "uploaded_at": metadata_response.get("LastModified", "").isoformat() if hasattr(metadata_response.get("LastModified", ""), 'isoformat') else str(metadata_response.get("LastModified", ""))
                        })
                    except Exception:
                        versions.append({
                            "version": version_name,
                            "s3_prefix": version_prefix,
                            "model_type": "unknown",
                            "metrics": {}
                        })
            
        except Exception as e:
            print(f"❌ Failed to list versions: {e}")
        
        # Sort by version
        versions.sort(key=lambda x: x["version"], reverse=True)
        return versions
    
    def sync(self) -> bool:
        """
        Sync local model to S3 if it's newer or different.
        
        Returns:
            True if upload performed, False otherwise
        """
        local_version = self._get_local_version()
        
        if not local_version:
            print("❌ No local model found to sync")
            return False
        
        # Check if version exists in S3
        versions = self.list_versions()
        existing_versions = [v["version"] for v in versions]
        
        if local_version in existing_versions:
            print(f"✅ Version {local_version} already in S3, no sync needed")
            return False
        
        print(f"📤 Syncing new version {local_version} to S3...")
        return self.upload_model(local_version)


# ==============================================================================
# CLI COMMANDS
# ==============================================================================

def cmd_upload(args):
    """Upload model to S3."""
    storage = S3ModelStorage()
    
    if not storage.is_connected():
        print("\n❌ Cannot connect to S3. Check your AWS credentials.")
        print("\nRequired environment variables:")
        print("  AWS_ACCESS_KEY_ID")
        print("  AWS_SECRET_ACCESS_KEY")
        print("  S3_BUCKET_NAME (optional, default: bcn-rental-models)")
        return 1
    
    success = storage.upload_model(
        version=args.version,
        force=args.force
    )
    
    return 0 if success else 1


def cmd_download(args):
    """Download model from S3."""
    storage = S3ModelStorage()
    
    if not storage.is_connected():
        print("\n❌ Cannot connect to S3. Check your AWS credentials.")
        return 1
    
    success = storage.download_model(version=args.version)
    
    if success:
        print("\n⚠️  Remember to restart the API to load the new model!")
    
    return 0 if success else 1


def cmd_list(args):
    """List versions in S3."""
    storage = S3ModelStorage()
    
    if not storage.is_connected():
        print("\n❌ Cannot connect to S3. Check your AWS credentials.")
        return 1
    
    versions = storage.list_versions()
    
    print("\n" + "="*70)
    print("📦 S3 MODEL VERSIONS")
    print(f"   Bucket: {storage.bucket_name}")
    print("="*70)
    
    if not versions:
        print("\n   No versions found in S3")
    else:
        print(f"\n{'Version':<20} {'Type':<15} {'R²':<8} {'RMSE':<12}")
        print("-"*70)
        
        for v in versions:
            metrics = v.get("metrics", {})
            r2 = metrics.get("r2", "N/A")
            rmse = metrics.get("rmse", "N/A")
            
            r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
            rmse_str = f"€{rmse:,.0f}" if isinstance(rmse, (int, float)) else str(rmse)
            
            print(f"  {v['version']:<18} {v.get('model_type', 'N/A'):<15} "
                  f"{r2_str:<8} {rmse_str:<12}")
    
    print("="*70 + "\n")
    return 0


def cmd_sync(args):
    """Sync local to S3."""
    storage = S3ModelStorage()
    
    if not storage.is_connected():
        print("\n❌ Cannot connect to S3. Check your AWS credentials.")
        return 1
    
    success = storage.sync()
    return 0 if success else 1


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="S3 storage for model artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  AWS_ACCESS_KEY_ID       AWS access key
  AWS_SECRET_ACCESS_KEY   AWS secret key
  AWS_REGION              AWS region (default: eu-west-1)
  S3_BUCKET_NAME          S3 bucket name (default: bcn-rental-models)

Examples:
  python scripts/s3_storage.py upload              # Upload current model
  python scripts/s3_storage.py upload --force      # Overwrite existing
  python scripts/s3_storage.py download            # Download latest
  python scripts/s3_storage.py download -v 1.0.0   # Download specific
  python scripts/s3_storage.py list                # List all versions
  python scripts/s3_storage.py sync                # Sync if newer
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload model to S3")
    upload_parser.add_argument("-v", "--version", help="Version tag (defaults to local version)")
    upload_parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing version")
    upload_parser.set_defaults(func=cmd_upload)
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download model from S3")
    download_parser.add_argument("-v", "--version", default="latest", help="Version to download")
    download_parser.set_defaults(func=cmd_download)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List versions in S3")
    list_parser.set_defaults(func=cmd_list)
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync local model to S3")
    sync_parser.set_defaults(func=cmd_sync)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
