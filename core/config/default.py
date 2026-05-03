VERSION = "v2.12.0"

DEFAULT_CONFIG = {
    "bot_config": {
        "bot": {
            "max_memory_length": 10,
            "max_message_interval": 2,
            "max_buffer_messages": 5,
            "min_message_delay": 2,
            "max_message_delay": 5
        },
        "agent": {
            "max_tool_loop": 2
        },
        "selfie": {
            "path": None
        }
    },
    "locale": {
        "lang": None,
        "TZ": None
    },
    "providers": {},  # ID: Provider config dict
    "models": {
        "default_llm": None,  # Provider ID - Model ID
        "default_fast_llm": None,
        "default_vlm": None,
        "default_tts": None,
        "default_stt": None,
        "default_image": None,
        "default_embedding": None,
        "default_rerank": None,
        "default_video": None,
        # Phase 0.1: when set, the main chat path resolves through ModelGroupManager
        # instead of calling provider_mgr.get_default_llm() directly. Empty string = use
        # the legacy path (single-model via models.default_llm). Group definitions live
        # in data/config/model_groups.json.
        "default_llm_group": "",
        "default_fast_llm_group": "",
    },
    "adapters": {},  # ID: Adapter config dict
    "telemetry": {
        "enabled": True,
        "client_uuid": None,
        "secret_key": None
    },
    "database": {
        "url": None,
        "echo": False
    }
}
