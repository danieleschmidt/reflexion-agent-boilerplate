"""Framework adapters for reflexion integration."""

from .autogen import AutoGenReflexion, create_reflexive_autogen_agent
from .crewai import ReflexiveCrewMember, ReflexiveCrew
from .langchain import ReflexionChain, ReflexionLangChainAgent, LangChainReflexionMiddleware

__all__ = [
    "AutoGenReflexion", 
    "create_reflexive_autogen_agent",
    "ReflexiveCrewMember",
    "ReflexiveCrew", 
    "ReflexionChain",
    "ReflexionLangChainAgent",
    "LangChainReflexionMiddleware"
]