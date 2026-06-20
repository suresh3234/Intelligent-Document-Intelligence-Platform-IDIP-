import os
import json
import logging
import subprocess
from typing import Optional
import pandas as pd

from config.settings import Settings

logger = logging.getLogger("idip.mlops.data_versioning")

def _run_dvc_command(args: list[str]) -> subprocess.CompletedProcess:
    """Helper to run a DVC command in the context of the active Poetry virtualenv."""
    # Find the absolute path to the project root directory
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # We invoke Python's DVC module runner, which is extremely robust across platforms
    cmd = ["poetry", "run", "python", "-m", "dvc"] + args
    logger.info(f"Running DVC command: {' '.join(cmd)} in CWD: {cwd}")
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        logger.error(f"DVC Command failed. Stdout: {result.stdout}, Stderr: {result.stderr}")
        raise RuntimeError(f"DVC Command failed with code {result.returncode}: {result.stderr}")
    return result

def init_dvc() -> None:
    """Initializes DVC inside the project directory if it hasn't been initialized already."""
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dvc_dir = os.path.join(cwd, ".dvc")
    if not os.path.exists(dvc_dir):
        logger.info("Initializing DVC with --no-scm configuration...")
        try:
            # We use --no-scm to prevent DVC from demanding a Git repository integration
            _run_dvc_command(["init", "--no-scm", "--f"])
        except Exception as e:
            logger.error(f"Failed to initialize DVC: {e}")
            raise

def setup_remote() -> None:
    """Registers default remote target pointing to the S3 bucket repository."""
    try:
        res = _run_dvc_command(["remote", "list"])
        if "idip_remote" not in res.stdout:
            logger.info("Registering default DVC remote pointing to s3://idip-dvc-store")
            _run_dvc_command(["remote", "add", "-d", "idip_remote", "s3://idip-dvc-store"])
    except Exception as e:
        logger.error(f"Failed to configure default DVC remote: {e}")
        # We do not crash on remote configuration issues to keep test and local setups working
        pass

def track_dataset(dataset_dir: str = "data/raw", version: Optional[str] = None) -> str:
    """Tracks raw dataset contents under DVC control, pushing updates to remote and saving version tags."""
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(cwd, dataset_dir)
    
    # Ensure directory is present
    os.makedirs(full_path, exist_ok=True)
    
    # DVC requires at least one file inside a directory to track it
    keep_file = os.path.join(full_path, ".keep")
    if not os.path.exists(keep_file):
        with open(keep_file, "w") as f:
            f.write("")

    init_dvc()
    setup_remote()

    logger.info(f"Staging dataset files under: {dataset_dir}")
    _run_dvc_command(["add", dataset_dir])

    try:
        logger.info("Syncing DVC tracked files to default remote storage")
        _run_dvc_command(["push"])
    except Exception as e:
        logger.warning(f"DVC remote sync skipped or failed: {e}")

    # Read and update version details registry
    versions_file = os.path.join(cwd, "data", "versions.json")
    os.makedirs(os.path.dirname(versions_file), exist_ok=True)
    
    versions_data = {}
    if os.path.exists(versions_file):
        try:
            with open(versions_file) as f:
                versions_data = json.load(f)
        except Exception:
            logger.warning("Existing versions.json could not be read. Resetting repository.")

    settings = Settings()
    env = settings.ENVIRONMENT

    if not version:
        # Increment version using patch numbering
        latest_tag = versions_data.get("latest", f"v1.0.0-{env}")
        base_tag = latest_tag.split("-")[0].lstrip("v")
        parts = base_tag.split(".")
        if len(parts) == 3:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            patch += 1
            version = f"v{major}.{minor}.{patch}-{env}"
        else:
            version = f"v1.0.1-{env}"
    else:
        # Format version: v{major}.{minor}.{patch}-{environment}
        if f"-{env}" not in version:
            version = f"{version}-{env}"

    versions_data[version] = {
        "dataset_dir": dataset_dir,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    versions_data["latest"] = version

    with open(versions_file, "w") as f:
        json.dump(versions_data, f, indent=2)

    logger.info(f"Dataset successfully versioned under tag: {version}")
    return version

def auto_version_on_batch(batch_size: int, dataset_dir: str = "data/raw") -> Optional[str]:
    """Automatically logs new dataset versions when the ingestion batch contains more than 1000 items."""
    if batch_size > 1000:
        logger.info(f"Ingested batch size of {batch_size} exceeds automated versioning limit of 1000.")
        return track_dataset(dataset_dir)
    logger.info(f"Ingested batch size of {batch_size} is within automated versioning limits.")
    return None
