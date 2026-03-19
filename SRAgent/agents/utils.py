import os
import re
import sys
import asyncio
from functools import wraps
from importlib import resources
from typing import Dict, Any, Optional, Set
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from dynaconf import Dynaconf
import openai


def load_settings() -> Dict[str, Any]:
    """
    Load settings from settings.yml file

    Args:
        env: Environment to load settings for ('test' or 'prod')

    Returns:
        Dictionary containing settings for the specified environment
    """
    # get path to settings
    if os.getenv("DYNACONF_SETTINGS_PATH"):
        s_path = os.getenv("DYNACONF_SETTINGS_PATH")
    else:
        s_path = str(resources.files("SRAgent").joinpath("settings.yml"))
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"Settings file not found: {s_path}")
    settings = Dynaconf(
        settings_files=[s_path], environments=True, env_switcher="DYNACONF"
    )
    return settings


def _get_agent_setting(
    settings: Dict[str, Any], key: str, agent_name: str, default: Any = None
) -> Any:
    """
    Get an agent-specific setting from Dynaconf-style settings.

    The setting may be a scalar value or a mapping keyed by agent name.
    """
    try:
        value = settings[key]
    except (KeyError, TypeError):
        return default

    if isinstance(value, dict):
        if agent_name in value:
            return value[agent_name]
        if "default" in value:
            return value["default"]
        return default

    try:
        return value[agent_name]
    except (KeyError, TypeError, AttributeError):
        try:
            return value["default"]
        except (KeyError, TypeError, AttributeError):
            return value if value is not None else default


def _get_provider(settings: Dict[str, Any], model_name: str, agent_name: str) -> str:
    """
    Determine which provider to use for the requested model.
    """
    if model_name.startswith("claude"):
        return "anthropic"

    provider = _get_agent_setting(settings, "provider", agent_name)
    if provider is None:
        return "openai"

    provider = str(provider).strip().lower()
    if provider == "azure":
        return "azure_openai"
    if provider == "claude":
        return "anthropic"
    return provider


def _get_azure_api_version(settings: Dict[str, Any], agent_name: str) -> str:
    """
    Resolve the Azure OpenAI API version from settings or environment.
    """
    configured_version = _get_agent_setting(
        settings, "azure_openai_api_version", agent_name
    )
    if configured_version:
        return configured_version

    return (
        os.getenv("AZURE_OPENAI_API_VERSION")
        or os.getenv("OPENAI_API_VERSION")
        or "2024-12-01-preview"
    )


def async_retry_on_flex_timeout(func):
    """
    Async decorator to retry with default tier if flex tier times out.
    """

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Check if we're using flex tier
        service_tier = getattr(self, "_service_tier", None)
        model_name = getattr(self, "model_name", None)

        if service_tier != "flex":
            # Not using flex tier, just call the function normally
            return await func(self, *args, **kwargs)

        try:
            # Try with flex tier first
            return await func(self, *args, **kwargs)
        except (asyncio.TimeoutError, openai.APITimeoutError) as e:
            print(
                f"Flex tier timeout for model {model_name}, retrying with standard tier...",
                file=sys.stderr,
            )

            # Create a new instance with default tier
            if hasattr(self, "_fallback_model"):
                # Use pre-created fallback model if available
                fallback_model = self._fallback_model
            else:
                # Create fallback model on the fly
                fallback_kwargs = {
                    "model_name": self.model_name,
                    "temperature": getattr(self, "temperature", None),
                    "max_tokens": getattr(self, "max_tokens", None),
                }
                # Add reasoning_effort if it's an o-model
                if hasattr(self, "reasoning_effort"):
                    fallback_kwargs["reasoning_effort"] = self.reasoning_effort
                    fallback_kwargs["temperature"] = None
                fallback_model = ChatOpenAI(**fallback_kwargs)

            # Retry with default tier
            return await fallback_model.ainvoke(*args, **kwargs)
        except Exception as e:
            # For other exceptions, just raise them
            raise

    return wrapper


def sync_retry_on_flex_timeout(func):
    """
    Sync decorator to retry with default tier if flex tier times out.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check if we're using flex tier
        service_tier = getattr(self, "_service_tier", None)
        model_name = getattr(self, "model_name", None)

        if service_tier != "flex":
            # Not using flex tier, just call the function normally
            return func(self, *args, **kwargs)

        try:
            # Try with flex tier first
            return func(self, *args, **kwargs)
        except (openai.APITimeoutError,) as e:
            print(
                f"Flex tier timeout for model {model_name}, retrying with standard tier...",
                file=sys.stderr,
            )

            # Create a new instance with default tier
            if hasattr(self, "_fallback_model"):
                # Use pre-created fallback model if available
                fallback_model = self._fallback_model
            else:
                # Create fallback model on the fly
                fallback_kwargs = {
                    "model_name": self.model_name,
                    "temperature": getattr(self, "temperature", None),
                    "max_tokens": getattr(self, "max_tokens", None),
                }
                # Add reasoning_effort if it's an o-model
                if hasattr(self, "reasoning_effort"):
                    fallback_kwargs["reasoning_effort"] = self.reasoning_effort
                    fallback_kwargs["temperature"] = None
                fallback_model = ChatOpenAI(**fallback_kwargs)

            # Retry with default tier
            return fallback_model.invoke(*args, **kwargs)
        except Exception as e:
            # For other exceptions, just raise them
            raise

    return wrapper


class FlexTierChatOpenAI(ChatOpenAI):
    """
    Extended ChatOpenAI that supports automatic fallback from flex to default tier.
    """

    def __init__(self, *args, service_tier: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._service_tier = service_tier

        # Create fallback model if using flex tier
        if service_tier == "flex":
            fallback_kwargs = kwargs.copy()
            fallback_kwargs.pop("service_tier", None)
            fallback_kwargs.pop("timeout", None)
            self._fallback_model = ChatOpenAI(**fallback_kwargs)

    @async_retry_on_flex_timeout
    async def ainvoke(self, *args, **kwargs):
        return await super().ainvoke(*args, **kwargs)

    @sync_retry_on_flex_timeout
    def invoke(self, *args, **kwargs):
        return super().invoke(*args, **kwargs)


def set_model(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    agent_name: str = "default",
    max_tokens: Optional[int] = None,
    service_tier: Optional[str] = None,
) -> Any:
    """
    Create a model instance with settings from configuration
    Args:
        model_name: Override model name from settings
        temperature: Override temperature from settings
        reasoning_effort: Override reasoning effort from settings
        agent_name: Name of the agent to get settings for
        max_tokens: Maximum number of tokens to use for the model
        service_tier: Service tier to use for the model
    Returns:
        Configured model instance
    """
    # Load settings
    settings = load_settings()

    # Use provided params or get from settings
    if model_name is None:
        model_name = _get_agent_setting(settings, "models", agent_name)
        if model_name is None:
            raise ValueError(f"No model name was provided for agent '{agent_name}'")

    if temperature is None:
        temperature = _get_agent_setting(settings, "temperature", agent_name)
        if temperature is None:
            raise ValueError(f"No temperature was provided for agent '{agent_name}'")
    if reasoning_effort is None:
        reasoning_effort = _get_agent_setting(
            settings, "reasoning_effort", agent_name
        )
        if reasoning_effort is None and temperature is None:
            raise ValueError(
                f"No reasoning effort or temperature was provided for agent '{agent_name}'"
            )
    if service_tier is None:
        service_tier = _get_agent_setting(
            settings, "service_tier", agent_name, default="default"
        )

    # Get timeout from settings (optional)
    timeout = _get_agent_setting(settings, "flex_timeout", agent_name, default=180.0)

    provider = _get_provider(settings, model_name=model_name, agent_name=agent_name)

    # Validate service_tier for OpenAI models
    if (
        provider == "openai"
        and service_tier == "flex"
        and not re.search(r"^(o[0-9]|^gpt-5)", model_name)
    ):
        raise ValueError(
            f"Service tier 'flex' only works with o3 and o4-mini, & gpt-5* models, not {model_name} (agent: {agent_name})"
        )

    # Agents that use structured outputs must not enable Claude "thinking" mode
    STRUCTURED_OUTPUT_AGENTS: Set[str] = {
        "convert_router",
        "metadata_router",
        "accessions",
        "get_entrez_ids",
        "entrez_convert",
        "metadata",
    }

    # Check model provider and initialize appropriate model
    if provider == "anthropic":  # e.g.,  "claude-3-7-sonnet-20250219"
        if agent_name in STRUCTURED_OUTPUT_AGENTS:
            think_tokens = 0
        elif reasoning_effort == "low":
            think_tokens = 1024
        elif reasoning_effort == "medium":
            think_tokens = 1024 * 2
        elif reasoning_effort == "high":
            think_tokens = 1024 * 4
        else:
            think_tokens = 0
        if think_tokens > 0:
            if not max_tokens:
                max_tokens = 1024
            max_tokens += think_tokens
            thinking = {"type": "enabled", "budget_tokens": think_tokens}
            temperature = None
        else:
            thinking = {"type": "disabled"}
            if temperature is None:
                raise ValueError(
                    "Temperature is required for Claude models if reasoning_effort is not set"
                )
        if not max_tokens:
            max_tokens = 1024
        model = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            thinking=thinking,
            max_tokens=max_tokens,
        )
    elif provider == "azure_openai":
        if service_tier == "flex":
            raise ValueError(
                "Service tier 'flex' is not supported for Azure OpenAI deployments"
            )

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT must be set when provider is azure_openai"
            )
        if not azure_api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY must be set when provider is azure_openai"
            )

        azure_model_name = _get_agent_setting(
            settings, "azure_openai_model", agent_name, default=model_name
        )
        azure_deployment = _get_agent_setting(
            settings, "azure_openai_deployment", agent_name, default=model_name
        )
        azure_model_version = _get_agent_setting(
            settings,
            "azure_openai_model_version",
            agent_name,
            default=os.getenv("AZURE_OPENAI_MODEL_VERSION"),
        )
        azure_api_version = _get_azure_api_version(settings, agent_name)

        azure_kwargs = {
            "azure_deployment": azure_deployment,
            "api_version": azure_api_version,
            "azure_endpoint": azure_endpoint,
            "api_key": azure_api_key,
            "model": azure_model_name,
        }
        if azure_model_version:
            azure_kwargs["model_version"] = azure_model_version
        if max_tokens is not None:
            azure_kwargs["max_tokens"] = max_tokens

        if azure_model_name.startswith("gpt-4"):
            azure_kwargs["temperature"] = temperature
        elif re.search(r"(^o[0-9]|^gpt-5)", azure_model_name):
            azure_kwargs["reasoning_effort"] = reasoning_effort
        else:
            raise ValueError(f"Model {azure_model_name} not supported")

        model = AzureChatOpenAI(**azure_kwargs)
    elif model_name.startswith("gpt-4"):
        # GPT-4o models use temperature but not reasoning_effort
        # Use FlexTierChatOpenAI for automatic fallback support
        model = FlexTierChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            reasoning_effort=None,
            max_tokens=max_tokens,
            service_tier=service_tier,
            timeout=timeout if service_tier == "flex" else None,
        )
    elif re.search(r"(^o[0-9]|^gpt-5)", model_name):
        # o[0-9] and gpt-5 models use reasoning_effort but not temperature
        # Use FlexTierChatOpenAI for automatic fallback support
        model = FlexTierChatOpenAI(
            model_name=model_name,
            temperature=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            service_tier=service_tier,
            timeout=timeout if service_tier == "flex" else None,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    return model


# main
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)

    # load settings
    settings = load_settings()
    print(settings)

    # set model
    model = set_model(model_name="claude-sonnet-4-5", agent_name="default")
    print(model)
