from src.config import _secret_env_pairs


def test_nested_secret_sections_map_to_env_names():
    secrets = {
        "retrieval": {"top_k": 5, "similarity_threshold": 0.40},
        "llm": {"provider": "ollama"},
        "ollama": {"api_key": "k", "base_url": "https://ollama.com/v1", "model": "gemma4:31b-cloud"},
        "nvidia": {"llm_model": "nvidia/x"},
    }

    pairs = dict(_secret_env_pairs(secrets))

    assert pairs["TOP_K"] == "5"
    assert pairs["SIMILARITY_THRESHOLD"] == "0.4"
    assert pairs["LLM_PROVIDER"] == "ollama"
    assert pairs["OLLAMA_API_KEY"] == "k"
    assert pairs["OLLAMA_MODEL"] == "gemma4:31b-cloud"
    assert pairs["NVIDIA_LLM_MODEL"] == "nvidia/x"


def test_flat_secrets_still_work():
    assert dict(_secret_env_pairs({"LLM_PROVIDER": "ollama"})) == {"LLM_PROVIDER": "ollama"}
