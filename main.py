from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from typing_extensions import Annotated
from prompts import (
    FLAGSHIP_PROMPT,
    FLAGSHIP_REFLECTION_PROMPT,
    SEASONAL_PROMPT,
    SEASONAL_CONTENT_PROMPT,
    EVERGREEN_PROMPT,
    Edit_PROMPT
)

# Define types for concurrent access
ConfigValue = Annotated[str, "config"]
MessageList = Annotated[List[dict], "messages"]

class ContentState(TypedDict):
    """State for the content brainstorm workflow"""
    flagship_messages: MessageList
    flagship_reflection_messages: MessageList
    seasonal_event_messages: MessageList
    seasonal_content_messages: MessageList
    evergreen_messages: MessageList
    editing_messages: MessageList
    core_value_provided: ConfigValue
    target_audience: ConfigValue
    monetization: ConfigValue
    persona: ConfigValue


class FlagshipAgent:
    def __init__(self, model):
        self.model = model

    def flagship(self, state: ContentState):
        """Provide a numbered list of 10 flagship content topics"""
        prompt = FLAGSHIP_PROMPT.format(
            core_value_provided=state["core_value_provided"],
            target_audience=state["target_audience"],
            persona=state["persona"],
            monetization=state["monetization"]
        )
        
        response = self.model.invoke(prompt)
        return {
            "flagship_messages": [{"role": "assistant", "content": response.content}]
        }
        
class FlagshipReflectionAgent:
    def __init__(self, model):
        self.model = model

    def flagship_reflection(self, state: ContentState):
        """select the top 5 flagship content topics and Propose a unique angle for each topic"""
        flagship_content = "\n".join([msg["content"] for msg in state.get("flagship_messages", [])])
        
        prompt = FLAGSHIP_REFLECTION_PROMPT.format(
            core_value_provided=state["core_value_provided"],
            target_audience=state["target_audience"],
            persona=state["persona"],
            monetization=state["monetization"],
            flagship_content=flagship_content
        )
        
        response = self.model.invoke(prompt)
        return {
            "flagship_reflection_messages": [{"role": "assistant", "content": response.content}]
        }
    
class SeasonalEventAgent:
    def __init__(self, model):
        self.model = model

    def seasonal_event(self, state: ContentState):
        """Identify Holidays and Seasonal Events"""
        prompt = SEASONAL_PROMPT.format(
            core_value_provided=state["core_value_provided"],
            target_audience=state["target_audience"],
            persona=state["persona"],
            monetization=state["monetization"]
        )
        
        response = self.model.invoke(prompt)
        return {
            "seasonal_event_messages": [{"role": "assistant", "content": response.content}]
        }
    
class SeasonalContentAgent:
    def __init__(self, model):
        self.model = model

    def seasonal_content(self, state: ContentState):
        """Suggest seasonal content and Propose a unique angle for each topic"""
        seasonal_events = "\n".join([msg["content"] for msg in state.get("seasonal_event_messages", [])])
        
        prompt = SEASONAL_CONTENT_PROMPT.format(
            core_value_provided=state["core_value_provided"],
            target_audience=state["target_audience"],
            persona=state["persona"],
            monetization=state["monetization"],
            seasonal_events=seasonal_events
        )
        
        response = self.model.invoke(prompt)
        return {
            "seasonal_content_messages": [{"role": "assistant", "content": response.content}]
        }
    
class EvergreenAgent:
    def __init__(self, model):
        self.model = model

    def evergreen(self, state: ContentState):
        """Suggest evergreen content that is always relevant no matter what year it is"""
        prompt = EVERGREEN_PROMPT.format(
            core_value_provided=state["core_value_provided"],
            target_audience=state["target_audience"],
            persona=state["persona"],
            monetization=state["monetization"]
        )
        
        response = self.model.invoke(prompt)
        return {
            "evergreen_messages": [{"role": "assistant", "content": response.content}]
        }
    
class EditingAgent:
    def __init__(self, model):
        self.model = model

    def editing(self, state: ContentState):
        """Summarize the content and generate a content report"""
        flagship_content = "\n".join([msg["content"] for msg in state.get("flagship_reflection_messages", [])])
        seasonal_content = "\n".join([msg["content"] for msg in state.get("seasonal_content_messages", [])])
        evergreen_content = "\n".join([msg["content"] for msg in state.get("evergreen_messages", [])])
        
        prompt = Edit_PROMPT.format(
            core_value_provided=state["core_value_provided"],
            target_audience=state["target_audience"],
            persona=state["persona"],
            monetization=state["monetization"],
            flagship_content=flagship_content,
            seasonal_content=seasonal_content,
            evergreen_content=evergreen_content
        )
        
        response = self.model.invoke(prompt)
        return {
            "editing_messages": [{"role": "assistant", "content": response.content}]
        }

def create_graph(model):
    """Creates and returns a compiled StateGraph with the given model"""
    # Create graph with properly typed state
    graph = StateGraph(ContentState)
    
    # Initialize agents
    flagship_agent = FlagshipAgent(model)
    flagship_reflection_agent = FlagshipReflectionAgent(model)
    seasonal_event_agent = SeasonalEventAgent(model)
    seasonal_content_agent = SeasonalContentAgent(model)
    evergreen_agent = EvergreenAgent(model)
    editing_agent = EditingAgent(model)
    
    # Add nodes
    graph.add_node("flagship", flagship_agent.flagship)
    graph.add_node("flagship_reflection", flagship_reflection_agent.flagship_reflection)
    graph.add_node("seasonal_event", seasonal_event_agent.seasonal_event)
    graph.add_node("seasonal_content", seasonal_content_agent.seasonal_content)
    graph.add_node("evergreen", evergreen_agent.evergreen)
    graph.add_node("editing", editing_agent.editing)

    # Set up conditional branching and edges
    graph.add_edge(START, "flagship")
    graph.add_edge(START, "seasonal_event")
    graph.add_edge(START, "evergreen")

    # Add sequential dependencies
    graph.add_edge("flagship", "flagship_reflection")
    graph.add_edge("flagship_reflection", "editing")
    graph.add_edge("seasonal_event", "seasonal_content")
    graph.add_edge("seasonal_content", "editing")
    graph.add_edge("evergreen", "editing")
    
    # Add final edge
    graph.add_edge("editing", END)
    
    return graph.compile()