import json
import os
import sys
import shutil # For backing up malformed config
import copy # For deepcopy

CONFIG_FILE_PATH = "config.json"

DEFAULT_CONFIG = {
    "cookies": {
        "GEMINI_SECURE_1PSID": None,
        "GEMINI_SECURE_1PSIDTS": None
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000
    },
    "gemini_settings": {
        "default_model": None
    }
}

def _deep_update(source, overrides):
    """
    Recursively update a dict with values from another dict.
    If a key in overrides is also in source and both are dicts,
    then recursively update. Otherwise, override the value in source.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(source.get(key), dict):
            source[key] = _deep_update(source.get(key, {}), value)
        else:
            source[key] = value
    return source

def load_config() -> dict:
    """
    Loads the configuration from config.json.
    If the file doesn't exist, a default one is created.
    If the file is malformed, it's backed up and a default one is created.
    Missing keys from the default config are added.
    """
    config_needs_saving = False
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, "r") as f:
                loaded_config = json.load(f)

            # Ensure all default keys are present (handles partial configs)
            # Create a deep copy of default config to fill in,
            # ensuring nested dicts like gemini_settings are also copied, not referenced.
            updated_config = copy.deepcopy(DEFAULT_CONFIG) # Changed to deepcopy
            _deep_update(updated_config, loaded_config) # Update default with loaded, preserving structure

            # If the loaded config was partial and got updated, or if _deep_update modified it
            # by adding new default keys (e.g. gemini_settings).
            if updated_config != loaded_config:
                 # A simple way to check if new structures were added is to compare after update.
                 # This might trigger more saves than strictly necessary if order changes,
                 # but ensures defaults are written if missing.
                if not all(k in loaded_config for k in DEFAULT_CONFIG.keys()) or \
                   not all(k in loaded_config.get("cookies", {}) for k in DEFAULT_CONFIG["cookies"].keys()) or \
                   not all(k in loaded_config.get("server", {}) for k in DEFAULT_CONFIG["server"].keys()) or \
                   not all(k in loaded_config.get("gemini_settings", {}) for k in DEFAULT_CONFIG["gemini_settings"].keys()):
                    print(f"Info: Configuration at '{CONFIG_FILE_PATH}' was updated with default values for missing keys.", file=sys.stderr)
                    config_needs_saving = True

            config = updated_config

        except json.JSONDecodeError:
            malformed_backup_path = CONFIG_FILE_PATH + ".malformed"
            print(f"Error: Malformed JSON in '{CONFIG_FILE_PATH}'. "
                  f"Backing up to '{malformed_backup_path}'. Creating new default config.", file=sys.stderr)
            try:
                shutil.copy(CONFIG_FILE_PATH, malformed_backup_path)
            except IOError as e:
                print(f"Error: Could not back up malformed config: {e}", file=sys.stderr)

            config = copy.deepcopy(DEFAULT_CONFIG) # Start fresh with a deepcopy
            config_needs_saving = True # Needs to be saved as it's a new default
            # Override with environment variables for cookies if creating a new default due to malformation
            config["cookies"]["GEMINI_SECURE_1PSID"] = os.getenv("GEMINI_SECURE_1PSID", config["cookies"]["GEMINI_SECURE_1PSID"])
            config["cookies"]["GEMINI_SECURE_1PSIDTS"] = os.getenv("GEMINI_SECURE_1PSIDTS", config["cookies"]["GEMINI_SECURE_1PSIDTS"])
            # default_model will remain None, as we don't source it from env for now

    else:
        print(f"Info: Configuration file '{CONFIG_FILE_PATH}' not found. Creating default config.", file=sys.stdout)
        config = copy.deepcopy(DEFAULT_CONFIG) # Use deepcopy
        # Override with environment variables for cookies when creating a new default
        env_psid = os.getenv("GEMINI_SECURE_1PSID")
        env_psidts = os.getenv("GEMINI_SECURE_1PSIDTS")

        if env_psid:
            config["cookies"]["GEMINI_SECURE_1PSID"] = env_psid
        if env_psidts:
            config["cookies"]["GEMINI_SECURE_1PSIDTS"] = env_psidts

        config_needs_saving = True

    if config_needs_saving:
        save_config(config)

    return config

def save_config(config_data: dict):
    """
    Saves the given configuration data to config.json.
    """
    try:
        with open(CONFIG_FILE_PATH, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"Info: Configuration saved to '{CONFIG_FILE_PATH}'.", file=sys.stdout)
    except IOError as e:
        print(f"Error: Could not write configuration to '{CONFIG_FILE_PATH}': {e}", file=sys.stderr)

if __name__ == '__main__':
    # Example usage:
    # This will load or create/load config, then print it, then save it (if modified or created).
    print("--- Initial load or create ---")
    cfg = load_config()
    print("Loaded config:", json.dumps(cfg, indent=2))

    # Example of modifying and saving
    # cfg["server"]["port"] = 8001
    # save_config(cfg)
    # print("\n--- Config after modification (if any) ---")
    # cfg_reloaded = load_config()
    # print("Reloaded config:", json.dumps(cfg_reloaded, indent=2))

    # Test malformed scenario (manual setup needed for this test)
    # 1. Create a malformed config.json
    # 2. Run script: python config_utils.py
    # 3. Check for .malformed file and new default config.json

    # Test partial config scenario (manual setup needed for this test)
    # 1. Create config.json with only e.g. {"server": {"port": 9000}}
    # 2. Run script.
    # 3. Check if cookies section was added from defaults.
    print("\n--- To test specific scenarios, modify config.json manually and re-run ---")
