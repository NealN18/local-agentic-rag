import yaml
from pathlib import Path
from urllib.parse import urlparse
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class DomainPolicy(BaseModel):
    fetcher: str = "static"
    respect_robots: bool = True
    rate_limit_rps: float = 1.0
    timeout: int = 20
    allowed_paths: list[str] = []
    denied_paths: list[str] = []

def load_policies(path: Path = Path("policies.yaml")) -> dict:
    if not path.exists():
        logger.warning(f"Policy file {path} not found. Using empty defaults.")
        return {"default": DomainPolicy().model_dump(), "domains": {}}
    
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
        
    return raw

def get_policy_for_url(url: str, policies: dict) -> DomainPolicy:
    domain = urlparse(url).netloc.lower()
    domain_config = policies.get("domains", {}).get(domain, {})
    default_config = policies.get("default", {})
    merged = {**default_config, **domain_config}
    return DomainPolicy(**merged)

def is_path_allowed(url: str, policy: DomainPolicy) -> bool:
    path = urlparse(url).path
    
    if any(path.startswith(denied) for denied in policy.denied_paths if denied):
        return False
        
    if policy.allowed_paths:
        return any(path.startswith(allowed) for allowed in policy.allowed_paths if allowed)
        
    return True