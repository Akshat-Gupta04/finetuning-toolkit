"""
Production-ready configuration manager with environment variable support
Handles secure loading of API keys, tokens, and configuration from .env files
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import warnings

from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
import yaml


class ConfigManager:
    """
    Production-ready configuration manager that handles:
    - Environment variables from .env files
    - YAML configuration files
    - Security and validation
    - Default values and overrides
    """

    def __init__(self, env_file: Optional[str] = None, project_root: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            env_file: Path to .env file (default: .env in project root)
            project_root: Project root directory (default: auto-detect)
        """
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        self.env_file = env_file or (self.project_root / ".env")

        # Load environment variables
        self._load_environment()

        # Setup logging
        self._setup_logging()

        # Validate critical settings
        self._validate_environment()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ConfigManager initialized with project root: {self.project_root}")

    def _find_project_root(self) -> Path:
        """Auto-detect project root by looking for key files"""
        current = Path.cwd()

        # Look for key project files
        key_files = [".env.example", "requirements.txt", "README.md", "pyproject.toml"]

        for parent in [current] + list(current.parents):
            if any((parent / key_file).exists() for key_file in key_files):
                return parent

        # Fallback to current directory
        return current

    def _load_environment(self):
        """Load environment variables from .env file"""
        if self.env_file.exists():
            load_dotenv(self.env_file, override=True)
            print(f"✅ Loaded environment from: {self.env_file}")
        else:
            print(f"⚠️  No .env file found at: {self.env_file}")
            print(f"💡 Copy .env.example to .env and configure your settings")

    def _setup_logging(self):
        """Setup logging configuration from environment"""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        verbose = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"

        # Configure logging format
        if verbose:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        else:
            log_format = "%(asctime)s - %(levelname)s - %(message)s"

        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format=log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Disable warnings if requested
        if os.getenv("DISABLE_WARNINGS", "false").lower() == "true":
            warnings.filterwarnings("ignore")

    def _validate_environment(self):
        """Validate critical environment settings"""
        # Check for required directories
        required_dirs = ["OUTPUT_DIR", "LOGGING_DIR", "CACHE_DIR"]
        for dir_var in required_dirs:
            dir_path = self.get_path(dir_var)
            dir_path.mkdir(parents=True, exist_ok=True)

        # Validate CUDA settings
        cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_devices:
            try:
                import torch
                if torch.cuda.is_available():
                    available_gpus = torch.cuda.device_count()
                    requested_gpus = [int(x.strip()) for x in cuda_devices.split(",")]
                    if max(requested_gpus) >= available_gpus:
                        print(f"⚠️  Warning: Requested GPU {max(requested_gpus)} but only {available_gpus} available")
            except ImportError:
                pass

    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get configuration value with environment variable support

        Args:
            key: Configuration key
            default: Default value if not found
            required: Whether the key is required

        Returns:
            Configuration value
        """
        value = os.getenv(key, default)

        if required and value is None:
            raise ValueError(f"Required configuration key '{key}' not found in environment")

        # Type conversion for common boolean values
        if isinstance(value, str):
            if value.lower() in ("true", "yes", "1", "on"):
                return True
            elif value.lower() in ("false", "no", "0", "off"):
                return False

        return value

    def get_path(self, key: str, default: Optional[str] = None) -> Path:
        """Get path configuration value, resolving relative to project root"""
        path_str = self.get(key, default)
        if path_str is None:
            raise ValueError(f"Path configuration key '{key}' not found")

        path = Path(path_str)
        if not path.is_absolute():
            path = self.project_root / path

        return path.resolve()

    def get_hf_token(self) -> Optional[str]:
        """Get Hugging Face token with validation"""
        token = self.get("HF_TOKEN")

        if not token or token == "your_huggingface_token_here":
            print("⚠️  Warning: No valid Hugging Face token found")
            print("💡 Set HF_TOKEN in your .env file for private model access")
            return None

        # Validate token format (basic check)
        if not token.startswith(("hf_", "hf-")):
            print("⚠️  Warning: Hugging Face token format may be invalid")

        return token

    def get_wandb_config(self) -> Dict[str, Any]:
        """Get Weights & Biases configuration"""
        return {
            "api_key": self.get("WANDB_API_KEY"),
            "project": self.get("WANDB_PROJECT", "diffusion-finetuning"),
            "entity": self.get("WANDB_ENTITY"),
            "mode": self.get("WANDB_MODE", "online"),
            "run_prefix": self.get("WANDB_RUN_PREFIX", "experiment"),
        }

    def setup_huggingface(self):
        """Setup Hugging Face environment"""
        # Set HF token
        hf_token = self.get_hf_token()
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # Set HF home directory
        hf_home = self.get("HF_HOME")
        if hf_home:
            os.environ["HF_HOME"] = hf_home

        # Set offline mode
        hf_offline = self.get("HF_HUB_OFFLINE", "false")
        os.environ["HF_HUB_OFFLINE"] = str(hf_offline).lower()

        # Trust remote code setting
        trust_remote_code = self.get("TRUST_REMOTE_CODE", "false")
        os.environ["TRUST_REMOTE_CODE"] = str(trust_remote_code).lower()

    def setup_pytorch(self):
        """Setup PyTorch environment"""
        # CUDA settings
        cuda_devices = self.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        # Memory allocation
        cuda_alloc_conf = self.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf

        # Tokenizers parallelism
        tokenizers_parallelism = self.get("TOKENIZERS_PARALLELISM", "false")
        os.environ["TOKENIZERS_PARALLELISM"] = str(tokenizers_parallelism).lower()

        # Performance settings
        omp_threads = self.get("OMP_NUM_THREADS", "1")
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)

        mkl_threads = self.get("MKL_NUM_THREADS", "1")
        os.environ["MKL_NUM_THREADS"] = str(mkl_threads)

    def setup_wandb(self):
        """Setup Weights & Biases environment"""
        wandb_config = self.get_wandb_config()

        if wandb_config["api_key"] and wandb_config["api_key"] != "your_wandb_api_key_here":
            os.environ["WANDB_API_KEY"] = wandb_config["api_key"]

        if wandb_config["entity"]:
            os.environ["WANDB_ENTITY"] = wandb_config["entity"]

        os.environ["WANDB_MODE"] = wandb_config["mode"]

    def load_yaml_config(self, config_path: Union[str, Path]) -> DictConfig:
        """
        Load YAML configuration file with environment variable substitution

        Args:
            config_path: Path to YAML configuration file

        Returns:
            OmegaConf configuration object
        """
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.project_root / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load YAML with environment variable substitution
        with open(config_path, 'r') as f:
            yaml_content = f.read()

        # Replace environment variables in YAML
        yaml_content = os.path.expandvars(yaml_content)

        # Parse YAML
        config_dict = yaml.safe_load(yaml_content)
        config = OmegaConf.create(config_dict)

        # Apply environment overrides
        self._apply_env_overrides(config)

        return config

    def _apply_env_overrides(self, config: DictConfig):
        """Apply environment variable overrides to configuration"""
        # Override common paths
        if "training" in config:
            if "output_dir" in config.training:
                config.training.output_dir = self.get("OUTPUT_DIR", config.training.output_dir)
            if "logging_dir" in config.training:
                config.training.logging_dir = self.get("LOGGING_DIR", config.training.logging_dir)
            if "cache_dir" in config.training:
                config.training.cache_dir = self.get("CACHE_DIR", config.training.cache_dir)
            if "mixed_precision" in config.training:
                config.training.mixed_precision = self.get("MIXED_PRECISION", config.training.mixed_precision)

        # Override model paths
        if "model" in config:
            if "cache_dir" in config.model:
                config.model.cache_dir = self.get("CACHE_DIR", config.model.cache_dir)
            if "pretrained_model_name_or_path" in config.model:
                if "wan2" in config.model.name.lower():
                    config.model.pretrained_model_name_or_path = self.get("WAN2_1_MODEL_PATH", config.model.pretrained_model_name_or_path)
                elif "flux" in config.model.name.lower():
                    config.model.pretrained_model_name_or_path = self.get("FLUX_MODEL_PATH", config.model.pretrained_model_name_or_path)

        # Override dataset paths
        if "dataset" in config:
            if "train_data_dir" in config.dataset:
                config.dataset.train_data_dir = self.get("DATA_DIR", config.dataset.train_data_dir)

        # Override W&B settings
        if "wandb" in config:
            wandb_config = self.get_wandb_config()
            if wandb_config["project"]:
                config.wandb.project_name = wandb_config["project"]
            if wandb_config["entity"]:
                config.wandb.entity = wandb_config["entity"]

    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model loading kwargs with authentication"""
        kwargs = {}

        # Add authentication token
        hf_token = self.get_hf_token()
        use_auth_token = self.get("USE_AUTH_TOKEN", "true").lower() == "true"

        if hf_token and use_auth_token:
            kwargs["token"] = hf_token
            kwargs["use_auth_token"] = True

        # Add trust remote code setting
        trust_remote_code = self.get("TRUST_REMOTE_CODE", "false").lower() == "true"
        kwargs["trust_remote_code"] = trust_remote_code

        return kwargs

    def setup_all(self):
        """Setup all environment configurations"""
        print("🔧 Setting up production environment...")

        self.setup_huggingface()
        self.setup_pytorch()
        self.setup_wandb()

        print("✅ Environment setup complete!")

    def print_config_summary(self):
        """Print configuration summary for debugging"""
        print("\n" + "="*60)
        print("📋 CONFIGURATION SUMMARY")
        print("="*60)

        print(f"Project Root: {self.project_root}")
        print(f"Environment File: {self.env_file}")
        print(f"Output Directory: {self.get_path('OUTPUT_DIR', './outputs')}")
        print(f"Cache Directory: {self.get_path('CACHE_DIR', './cache')}")
        print(f"Logging Directory: {self.get_path('LOGGING_DIR', './logs')}")

        print(f"\nHugging Face Token: {'✅ Set' if self.get_hf_token() else '❌ Not set'}")

        wandb_config = self.get_wandb_config()
        print(f"W&B API Key: {'✅ Set' if wandb_config['api_key'] and wandb_config['api_key'] != 'your_wandb_api_key_here' else '❌ Not set'}")
        print(f"W&B Project: {wandb_config['project']}")

        print(f"\nCUDA Devices: {self.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"Mixed Precision: {self.get('MIXED_PRECISION', 'bf16')}")
        print(f"Log Level: {self.get('LOG_LEVEL', 'INFO')}")

        print("="*60)


# Global configuration manager instance
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    return config_manager


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration file with environment variable support"""
    return config_manager.load_yaml_config(config_path)


def setup_environment():
    """Setup production environment"""
    config_manager.setup_all()


if __name__ == "__main__":
    # Test configuration manager
    config_manager.print_config_summary()

    # Test loading a config file
    try:
        config = load_config("config/wan2_1_config.yaml")
        print(f"\n✅ Successfully loaded config: {config.model.name}")
    except Exception as e:
        print(f"\n❌ Error loading config: {e}")
