"""Tests for FR8: startup validation — validate_config and load_and_validate (obsidian-17w.5)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sr2_spectre.config import (
    CircularExtendsError,
    StartupConfigError,
    load_and_validate,
    validate_config,
)
from sr2_spectre.path_resolution import ConfigPathError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


# ---------------------------------------------------------------------------
# validate_config — empty / fully valid
# ---------------------------------------------------------------------------

class TestValidateConfigValid:
    def test_empty_config_is_valid(self):
        """{} is a valid config — no required top-level keys."""
        errors = validate_config({})
        assert errors == []

    def test_full_valid_config_no_errors(self):
        """A config with agent, models, and pipeline sections — all valid — returns no errors."""
        config = {
            "agent": {"name": "spectre"},
            "models": {
                "default": {"model": "gpt-4o"},
                "fast": {"model": "gpt-4o-mini", "base_url": "https://example.com"},
            },
            "pipeline": {
                "layers": [
                    {
                        "name": "my_layer",
                        "resolvers": [{"type": "prompt"}],
                        "transformers": [{"type": "trim"}],
                    }
                ]
            },
        }
        errors = validate_config(config)
        assert errors == []

    def test_config_without_agent_section_is_valid(self):
        """Config with models and pipeline but no agent is valid."""
        config = {
            "models": {"default": {"model": "claude-3"}},
            "pipeline": {"layers": [{"name": "layer1"}]},
        }
        errors = validate_config(config)
        assert errors == []

    def test_config_with_agent_no_name_field_is_valid(self):
        """Agent section without a 'name' field is valid (name is optional)."""
        config = {"agent": {"max_tool_rounds": 5}}
        errors = validate_config(config)
        assert errors == []


# ---------------------------------------------------------------------------
# validate_config — agent.name
# ---------------------------------------------------------------------------

class TestValidateConfigAgentName:
    def test_agent_name_empty_string_is_error(self):
        """agent.name = '' must produce a validation error."""
        config = {"agent": {"name": ""}}
        errors = validate_config(config)
        assert len(errors) >= 1
        assert any("agent" in e.lower() or "name" in e.lower() for e in errors)

    def test_agent_name_nonempty_string_is_valid(self):
        """agent.name = 'spectre' must produce no error."""
        config = {"agent": {"name": "spectre"}}
        errors = validate_config(config)
        assert errors == []


# ---------------------------------------------------------------------------
# validate_config — models
# ---------------------------------------------------------------------------

class TestValidateConfigModels:
    def test_models_must_be_dict(self):
        """models: [list] is invalid — must be a dict."""
        config = {"models": ["default"]}
        errors = validate_config(config)
        assert len(errors) >= 1
        assert any("models" in e.lower() for e in errors)

    def test_models_entry_missing_model_field(self):
        """An entry in models without a 'model' field is invalid."""
        config = {"models": {"default": {"base_url": "https://example.com"}}}
        errors = validate_config(config)
        assert len(errors) >= 1
        assert any("model" in e.lower() or "default" in e.lower() for e in errors)

    def test_models_entry_empty_model_field(self):
        """An entry in models with model='' is invalid."""
        config = {"models": {"default": {"model": ""}}}
        errors = validate_config(config)
        assert len(errors) >= 1
        assert any("model" in e.lower() or "default" in e.lower() for e in errors)

    def test_models_entry_valid(self):
        """A models entry with a non-empty 'model' string is valid."""
        config = {"models": {"default": {"model": "claude-3-opus"}}}
        errors = validate_config(config)
        assert errors == []

    def test_multiple_models_one_invalid_returns_error(self):
        """Multiple models where one is missing 'model' returns at least one error."""
        config = {
            "models": {
                "good": {"model": "gpt-4o"},
                "bad": {"base_url": "https://example.com"},
            }
        }
        errors = validate_config(config)
        assert len(errors) >= 1
        assert any("bad" in e.lower() or "model" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# validate_config — pipeline.layers
# ---------------------------------------------------------------------------

class TestValidateConfigPipeline:
    def test_pipeline_layer_missing_name_is_error(self):
        """A layer in pipeline.layers without a 'name' field is invalid."""
        config = {
            "pipeline": {
                "layers": [
                    {"resolvers": [{"type": "prompt"}]},
                ]
            }
        }
        errors = validate_config(config)
        assert len(errors) >= 1
        assert any("layer" in e.lower() or "name" in e.lower() for e in errors)

    def test_pipeline_layer_empty_name_is_error(self):
        """A layer in pipeline.layers with name='' is invalid."""
        config = {
            "pipeline": {
                "layers": [
                    {"name": ""},
                ]
            }
        }
        errors = validate_config(config)
        assert len(errors) >= 1
        assert any("layer" in e.lower() or "name" in e.lower() for e in errors)

    def test_pipeline_layer_valid_name_is_valid(self):
        """A layer with a non-empty 'name' string is valid."""
        config = {
            "pipeline": {
                "layers": [
                    {"name": "my_layer"},
                ]
            }
        }
        errors = validate_config(config)
        assert errors == []

    def test_pipeline_layer_resolver_not_dict_is_error(self):
        """A resolver that is not a dict (e.g. a string) is invalid."""
        config = {
            "pipeline": {
                "layers": [
                    {"name": "layer1", "resolvers": ["prompt_string"]},
                ]
            }
        }
        errors = validate_config(config)
        assert len(errors) >= 1

    def test_pipeline_layer_transformer_not_dict_is_error(self):
        """A transformer that is not a dict is invalid."""
        config = {
            "pipeline": {
                "layers": [
                    {"name": "layer1", "transformers": [42]},
                ]
            }
        }
        errors = validate_config(config)
        assert len(errors) >= 1

    def test_pipeline_no_layers_key_is_valid(self):
        """pipeline section without 'layers' key is valid."""
        config = {"pipeline": {"system_prompt": "you are helpful"}}
        errors = validate_config(config)
        assert errors == []


# ---------------------------------------------------------------------------
# validate_config — multiple errors aggregated
# ---------------------------------------------------------------------------

class TestValidateConfigMultipleErrors:
    def test_multiple_errors_all_returned(self):
        """When multiple validation errors exist, all are returned in the list."""
        config = {
            "agent": {"name": ""},                     # error: empty name
            "models": {"default": {"base_url": "x"}},  # error: missing model field
            "pipeline": {
                "layers": [{"resolvers": []}],          # error: layer missing name
            },
        }
        errors = validate_config(config)
        assert len(errors) >= 3

    def test_returns_list_type(self):
        """validate_config always returns a list, even for a valid config."""
        result = validate_config({})
        assert isinstance(result, list)

        result = validate_config({"agent": {"name": ""}})
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# load_and_validate — happy path
# ---------------------------------------------------------------------------

class TestLoadAndValidateHappyPath:
    def test_valid_config_files_returns_merged_dict(self, tmp_path):
        """Valid config files → returns the merged dict with no exception."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": "spectre"},
            "models": {"default": {"model": "gpt-4o"}},
        })

        result = load_and_validate(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert isinstance(result, dict)
        assert result["agent"]["name"] == "spectre"
        assert result["models"]["default"]["model"] == "gpt-4o"

    def test_no_config_files_empty_dict_returned(self, tmp_path):
        """No config files at any tier → returns {} (empty is valid)."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        result = load_and_validate(cwd=cwd, env={"SR2_HOME": str(sr2_home)})
        assert result == {}


# ---------------------------------------------------------------------------
# load_and_validate — structural errors propagate directly
# ---------------------------------------------------------------------------

class TestLoadAndValidateStructuralErrors:
    def test_circular_extends_propagates_directly(self, tmp_path):
        """CircularExtendsError from load_merged_config propagates, not wrapped."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        # Create a circular chain: .spectre.yaml extends itself
        spectre_file = cwd / ".spectre.yaml"
        _write_yaml(spectre_file, {"extends": ".spectre.yaml", "key": "value"})

        with pytest.raises(CircularExtendsError):
            load_and_validate(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

    def test_unset_env_var_in_extends_raises_config_path_error(self, tmp_path):
        """${UNSET_VAR} in extends path raises ConfigPathError, not StartupConfigError."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "extends": "${UNSET_SPECTRE_VAR}/base.yaml",
            "key": "value",
        })

        # env does NOT contain UNSET_SPECTRE_VAR
        with pytest.raises(ConfigPathError):
            load_and_validate(cwd=cwd, env={"SR2_HOME": str(sr2_home)})


# ---------------------------------------------------------------------------
# load_and_validate — validation errors raise StartupConfigError
# ---------------------------------------------------------------------------

class TestLoadAndValidateStartupConfigError:
    def test_invalid_config_raises_startup_config_error(self, tmp_path):
        """Merged config with validation errors raises StartupConfigError."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": ""},   # invalid: empty name
        })

        with pytest.raises(StartupConfigError):
            load_and_validate(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

    def test_startup_config_error_contains_all_errors(self, tmp_path):
        """StartupConfigError.errors contains all validation errors (not just the first)."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": ""},                     # error 1
            "models": {"default": {"base_url": "x"}},  # error 2: missing model
        })

        with pytest.raises(StartupConfigError) as exc_info:
            load_and_validate(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        e = exc_info.value
        assert isinstance(e.errors, list)
        assert len(e.errors) >= 2

    def test_startup_config_error_str_contains_error_text(self, tmp_path):
        """str(StartupConfigError) includes the validation error messages."""
        sr2_home = tmp_path / "sr2home"
        sr2_home.mkdir()
        cwd = tmp_path / "project"
        cwd.mkdir()

        _write_yaml(cwd / ".spectre.yaml", {
            "agent": {"name": ""},
        })

        with pytest.raises(StartupConfigError) as exc_info:
            load_and_validate(cwd=cwd, env={"SR2_HOME": str(sr2_home)})

        error_str = str(exc_info.value)
        assert len(error_str) > 0
        # The string should contain text from the errors list
        e = exc_info.value
        for error in e.errors:
            assert error in error_str


# ---------------------------------------------------------------------------
# StartupConfigError — direct construction
# ---------------------------------------------------------------------------

class TestStartupConfigError:
    def test_errors_attribute_preserved(self):
        """StartupConfigError.errors holds the list passed at construction."""
        errs = ["missing model field in 'default'", "agent.name is empty"]
        exc = StartupConfigError(errs)
        assert exc.errors == errs

    def test_str_joins_all_errors(self):
        """str(StartupConfigError) contains all error strings."""
        errs = ["error one", "error two"]
        exc = StartupConfigError(errs)
        msg = str(exc)
        assert "error one" in msg
        assert "error two" in msg

    def test_empty_errors_list_allowed(self):
        """StartupConfigError can be constructed with an empty list (edge case)."""
        exc = StartupConfigError([])
        assert exc.errors == []
