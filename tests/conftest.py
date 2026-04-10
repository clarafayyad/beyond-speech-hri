"""
Conftest: stub out heavy runtime dependencies (hardware SDKs, ML models, etc.)
so that unit tests can import interaction and agent modules without needing the
full robot/hardware stack installed.
"""

import sys
from unittest.mock import MagicMock

# Modules to stub out (only if they aren't already in sys.modules).
_STUBS = [
    # Image / display
    "PIL",
    "PIL.Image",
    # Social Interaction Cloud framework
    "sic_framework",
    "sic_framework.core",
    "sic_framework.core.sic_logging",
    "sic_framework.core.message_python2",
    "sic_framework.core.sic_application",
    "sic_framework.devices",
    "sic_framework.devices.device",
    "sic_framework.devices.desktop",
    "sic_framework.devices.naoqi_shared",
    "sic_framework.devices.common_naoqi",
    "sic_framework.devices.common_naoqi.naoqi_motion",
    "sic_framework.devices.common_naoqi.naoqi_text_to_speech",
    "sic_framework.services",
    "sic_framework.services.dialogflow",
    "sic_framework.services.dialogflow.dialogflow",
    "sic_framework.services.llm",
    # Numeric / audio / ML runtime
    "numpy",
    "soundfile",
    "sounddevice",
    "openai",
    "dotenv",
    "whisper",
    # Flask (game state server)
    "flask",
    # Agents
    "agents.dialog_manager",
    "agents.llm_agent",
    "agents.pepper_tablet",
    "agents.pepper_tablet.display_service",
    "agents.stt_manager",
    "agents.tts_manager",
    # Audio pipeline internals
    "multimodal_perception.audio",
    "multimodal_perception.audio.recorder",
    "multimodal_perception.audio.transcribe_audio",
    "multimodal_perception.audio.important_feature_extractor",
    "interaction.audio_pipeline",
]

for _mod in _STUBS:
    sys.modules.setdefault(_mod, MagicMock())

# The confidence classifier defines string constants that are used directly in
# production code and tests, so we expose the real values rather than a MagicMock.
_confidence_stub = MagicMock()
_confidence_stub.CONFIDENCE_LOW = "low"
_confidence_stub.CONFIDENCE_MEDIUM = "medium"
_confidence_stub.CONFIDENCE_HIGH = "high"
sys.modules.setdefault("multimodal_perception", MagicMock())
sys.modules.setdefault("multimodal_perception.model", MagicMock())
sys.modules["multimodal_perception.model.confidence_classifier"] = _confidence_stub
