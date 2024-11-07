import streamlit as st
from langchain_groq import ChatGroq
from main import create_graph
from langchain.callbacks.base import BaseCallbackHandler
from typing import TypedDict, List, Dict, Any
import os

# Define type structures
class Message(TypedDict):
    role: str
    content: str

class AppState(TypedDict):
    flagship_messages: List[Message]
    flagship_reflection_messages: List[Message]
    seasonal_event_messages: List[Message]
    seasonal_content_messages: List[Message]
    evergreen_messages: List[Message]
    editing_messages: List[Message]
    core_value_provided: str
    target_audience: str
    persona: str
    monetization: str

# Environment setup
def setup_environment():
    """Set up environment variables from Streamlit secrets"""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]["API_KEY"]
    os.environ["LANGCHAIN_PROJECT"] = "STATIC_CONTENT_MASTER"

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming output to Streamlit"""
    def __init__(self, container):
        self.container = container
        
    def on_llm_new_token(self, token: str, **kwargs):
        """Handle streaming tokens"""
        self.container.markdown(token)

def setup_page():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="Static Content Master",
        page_icon="🎯",
        layout="wide"
    )

def create_sidebar():
    """Create and configure the sidebar"""
    st.sidebar.header("🎯 Static Content Master")
    st.sidebar.markdown(
        "This app suggests Static Content that aligns with your social media positioning. "
        "To use this App, you need to provide a Groq API key, which you can get [here](https://console.groq.com/keys) for free."
    )
    
    st.sidebar.write("### Instructions")
    instructions = [
        "1️⃣ Enter your core value proposition",
        "2️⃣ Define your target audience",
        "3️⃣ Describe your brand persona",
        "4️⃣ Explain your monetization strategy",
        "5️⃣ Click 'Generate Suggestions' for detailed insights"
    ]
    for instruction in instructions:
        st.sidebar.write(instruction)
    
    try:
        st.sidebar.image("assets/logo01.jpg", use_column_width=True)
    except FileNotFoundError:
        st.sidebar.warning("Logo file not found. Please check the assets directory.")

def get_api_key() -> str | None:
    """Get and validate the Groq API key"""
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Enter your Groq API key...",
        help="Your key will not be stored"
    )
    
    if not api_key:
        st.info("Please add your Groq API key to continue.", icon="🔑")
        return None
    return api_key

def create_input_form() -> Dict[str, str] | None:
    """Create and handle the input form"""
    with st.form("positioning_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            core_value = st.text_area(
                "Core Value Proposition",
                placeholder="e.g., Easy-to-use AI tools for automating basic business processes",
                help="What primary value or benefit does your business offer?"
            )
            
            target_audience = st.text_area(
                "Target Audience",
                placeholder="e.g., Small business owners with limited technical knowledge",
                help="Who are you trying to reach and serve?"
            )
            
        with col2:
            persona = st.text_area(
                "Brand Persona",
                placeholder="e.g., A helpful, approachable AI consultant",
                help="What character or image does your brand project?"
            )
            
            monetization = st.text_area(
                "Monetization Strategy",
                placeholder="e.g., Affordable AI software-as-a-service subscriptions",
                help="How do you generate revenue?"
            )
        
        submit_button = st.form_submit_button("Generate Suggestions")
        
        if submit_button:
            if not all([core_value, target_audience, persona, monetization]):
                st.error("Please fill in all fields before generating suggestions.")
                return None
            
            return {
                "core_value": core_value,
                "target_audience": target_audience,
                "persona": persona,
                "monetization": monetization
            }
        return None

def create_initial_state(input_data: Dict[str, str]) -> AppState:
    """Create the initial state with proper typing"""
    return {
        "flagship_messages": [],
        "flagship_reflection_messages": [],
        "seasonal_event_messages": [],
        "seasonal_content_messages": [],
        "evergreen_messages": [],
        "editing_messages": [],
        "core_value_provided": input_data["core_value"],
        "target_audience": input_data["target_audience"],
        "persona": input_data["persona"],
        "monetization": input_data["monetization"]
    }

def generate_suggestions(api_key: str, input_data: Dict[str, str]) -> List[Message] | None:
    """Generate suggestions using the ChatGroq model"""
    try:
        chat_groq = ChatGroq(api_key=api_key, model="llama-3.2-90b-text-preview", temperature=0.2)
           
        # Create a placeholder for streaming output
        output_container = st.empty()
        
        # Initialize the callback handler
        callback_handler = StreamlitCallbackHandler(output_container)
        
        # Create the graph with the model
        graph = create_graph(chat_groq)
        
        # Create properly typed initial state
        initial_state = create_initial_state(input_data)
        
        # Execute the graph
        with st.spinner("Generating suggestions..."):
            final_state = graph.invoke(initial_state)
            
        st.success("Suggestions generated successfully!")
        
        # Collect all messages from different categories
        all_messages = []
        
        # Add flagship content
        if final_state.get("flagship_messages"):
            all_messages.append({"role": "assistant", "content": "## Flagship Content Ideas"})
            all_messages.extend(final_state["flagship_messages"])
        
        # Add flagship reflection
        if final_state.get("flagship_reflection_messages"):
            all_messages.append({"role": "assistant", "content": "## Top Flagship Content Analysis"})
            all_messages.extend(final_state["flagship_reflection_messages"])
        
        # Add seasonal events
        if final_state.get("seasonal_event_messages"):
            all_messages.append({"role": "assistant", "content": "## Seasonal Events"})
            all_messages.extend(final_state["seasonal_event_messages"])
        
        # Add seasonal content
        if final_state.get("seasonal_content_messages"):
            all_messages.append({"role": "assistant", "content": "## Seasonal Content Ideas"})
            all_messages.extend(final_state["seasonal_content_messages"])
        
        # Add evergreen content
        if final_state.get("evergreen_messages"):
            all_messages.append({"role": "assistant", "content": "## Evergreen Content Ideas"})
            all_messages.extend(final_state["evergreen_messages"])
        
        # Add final editing summary
        if final_state.get("editing_messages"):
            all_messages.append({"role": "assistant", "content": "## Content Strategy Summary"})
            all_messages.extend(final_state["editing_messages"])
        
        return all_messages
        
    except Exception as e:
        st.error(f"An error occurred while generating suggestions: {str(e)}")
        return None

def main():
    """Main application function"""
    try:
        setup_environment()
        setup_page()
        create_sidebar()
        
        api_key = get_api_key()
        if not api_key:
            return
        
        input_data = create_input_form()
        if input_data:
            suggestions = generate_suggestions(api_key, input_data)
            if suggestions:
                # Create tabs for different sections
                tab_titles = ["All Content", "Flagship", "Seasonal", "Evergreen", "Summary"]
                tabs = st.tabs(tab_titles)

                # Helper function to display messages with proper formatting
                def display_messages(messages):
                    for msg in messages:
                        st.markdown(msg["content"])
                        if not msg["content"].startswith("#"):  # Don't add dividers after headers
                            st.divider()

                # Helper function to extract content between section headers
                def extract_section_content(messages, start_headers, end_headers):
                    content = []
                    collecting = False
                    
                    for msg in messages:
                        if msg["role"] == "assistant":
                            # Check if we've hit a start header
                            if any(header in msg["content"] for header in start_headers):
                                collecting = True
                                content.append(msg)
                                continue
                            
                            # Check if we've hit an end header
                            if any(header in msg["content"] for header in end_headers):
                                collecting = False
                                continue
                            
                            # Collect content if we're in the right section
                            if collecting:
                                content.append(msg)
                    
                    return content
                
                with tabs[0]:  # All Content
                    display_messages(suggestions)
                
                with tabs[1]:  # Flagship
                    flagship_content = extract_section_content(
                        suggestions,
                        ["## Flagship Content Ideas", "## Top Flagship Content Analysis"],
                        ["## Seasonal Events", "## Seasonal Content Ideas", "## Evergreen Content Ideas", "## Content Strategy Summary"]
                    )
                    display_messages(flagship_content)
                
                with tabs[2]:  # Seasonal
                    seasonal_content = extract_section_content(
                        suggestions,
                        ["## Seasonal Events", "## Seasonal Content Ideas"],
                        ["## Evergreen Content Ideas", "## Content Strategy Summary"]
                    )
                    display_messages(seasonal_content)
                
                with tabs[3]:  # Evergreen
                    evergreen_content = extract_section_content(
                        suggestions,
                        ["## Evergreen Content Ideas"],
                        ["## Content Strategy Summary"]
                    )
                    display_messages(evergreen_content)
                
                with tabs[4]:  # Summary
                    summary_content = extract_section_content(
                        suggestions,
                        ["## Content Strategy Summary"],
                        ["## END"]  # Using a non-existent header to capture all remaining content
                    )
                    display_messages(summary_content)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
