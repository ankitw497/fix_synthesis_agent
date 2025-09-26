"""
Agent 3 (Synthesis · Viz · Deck) - Credit Portfolio Analysis Pipeline
"""

__version__ = "1.0.0"

from synthesis_agent.config import SynthesisConfig, load_config
from synthesis_agent.main import main

__all__ = [
    'SynthesisConfig',
    'load_config',
    'main'
]