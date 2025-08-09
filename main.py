#debate bot

from groq import Groq
import json
from dotenv import load_dotenv
import os
import re
import streamlit as st
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Debate Arena",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .debate-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .llm1-card {
        background: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .llm2-card {
        background: #f3e5f5;
        border-left: 5px solid #9c27b0;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .judge-card {
        background: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        min-width: 100px;
    }
    
    .round-header {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
    }
    
    .side-panel {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .debate-card,
    .llm1-card,
    .llm2-card,
    .judge-card,
    .stat-box,
    .side-panel {
        color: #000 !important;
    }
    .debate-card h3, .debate-card p,
    .llm1-card h4, .llm1-card h5, .llm1-card p,
    .llm2-card h4, .llm2-card h5, .llm2-card p,
    .judge-card h4, .judge-card p,
    .stat-box h3, .stat-box p,
    .side-panel h3, .side-panel h4, .side-panel p {
        color: #000 !important;
    }
</style>
""", unsafe_allow_html=True)

groq = Groq(api_key=st.secrets["GROQ_API_KEY"])

def goal_refinement_and_assignment(input: str):
    usr_goal = input
    response = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": f"""
You are a debate moderator responsible for facilitating structured intellectual discourse between two AI systems. Your task is to analyze the user's request, clarify the debate topic, and establish clear roles for two LLM participants.
Instructions:
Respond ONLY with valid JSON in a single code block, no explanations.
1. Goal Refinement: Transform the user's input into a clear, specific, and debatable proposition. Ensure the refined goal:
   - Is unambiguous and well-defined
   - Presents a meaningful point of contention
   - Allows for substantive arguments on multiple sides
   - Is neither too broad nor too narrow for productive debate

2. Role Assignment: Assign distinct, balanced roles to two LLM debaters that:
   - Create meaningful opposition or complementary perspectives
   - Are roughly equal in argumentative potential
   - Encourage thorough exploration of the topic
   - Avoid obviously advantaged or disadvantaged positions

3. Output Format: Return your response as valid JSON with the following structure:
   {{
     "refined_goal": "A clear, specific statement of what will be debated",
     "debaters": {{
       "llm_1": "First debater identifier (do not use this in roles, insted just use llm_1 or llm_2 insted)",
       "llm_2": "Second debater identifier"
     }},
     "roles": {{
       "llm_1": "Detailed description of LLM 1's position, perspective, and argumentative focus",
       "llm_2": "Detailed description of LLM 2's position, perspective, and argumentative focus"
     }}
   }}

Quality Standards:
- Ensure roles are substantive and provide clear guidance for argumentation
- Make the refined goal actionable and focused
- Balance the debate setup to promote fair and engaging discourse
- Consider multiple dimensions of the topic when assigning roles
"""},
            {"role": "user", "content": f"user goal: {usr_goal}"},
        ],
    )

    assignment = response.choices[0].message.content
    # Print the raw response for debugging
    print("Raw LLM response:\n", assignment)

    # Ensure assignment is a string before using re.search
    if assignment is None:
        raise ValueError("No content returned from LLM response.")

    # Extract JSON block from the response
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", assignment, re.DOTALL)
    if not json_match:
        # Try to find any JSON block (even without ```json)
        json_match = re.search(r"(\{[\s\S]*\})", assignment)
    if json_match:
        assignment_json_str = json_match.group(1)
    else:
        raise ValueError("No JSON object found in the LLM response.")

    try:
        assignment_json = json.loads(assignment_json_str)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON. Extracted string was:\n", assignment_json_str)
        raise ValueError("The response is not valid JSON. Please check the input and try again.") from e

    goal_of_llm1 = assignment_json["roles"]["llm_1"]
    goal_of_llm2 = assignment_json["roles"]["llm_2"]
    return goal_of_llm1, goal_of_llm2, assignment_json

def run_debate():
    global chat_history
    # LLM 1 responds to the latest message (user or LLM 2)
    llm_1_input = chat_history[-1]["content"] if chat_history else user_goal
    llm_1_response = llm_1_call(llm_1_input, goal_of_llm1)
    chat_history.append({"role": "llm_1", "content": llm_1_response})

    # LLM 2 responds to the latest message (LLM 1)
    llm_2_input = chat_history[-1]["content"]
    llm_2_response = llm_2_call(llm_2_input, goal_of_llm2)
    chat_history.append({"role": "llm_2", "content": llm_2_response})

    return {
        "llm_1_response": llm_1_response,
        "llm_2_response": llm_2_response
    }

def format_history_for_api(history):
    mapped = []
    for msg in history:
        if msg["role"] in ("llm_1", "llm_2"):
            mapped.append({"role": "assistant", "content": msg["content"]})
        else:
            mapped.append(msg)
    return mapped

def llm_1_call(input: str, goal_of_llm1: str):
    global chat_history
    recent_history = chat_history[-4:] if len(chat_history) >= 4 else chat_history
    api_history = format_history_for_api(recent_history)
    response = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": f"You are a debate bot. Your name: llm_1. Goal/stance description: {goal_of_llm1}"},
            *api_history,
            {"role": "user", "content": input},
        ],
    )
    return response.choices[0].message.content

def llm_2_call(input: str, goal_of_llm2: str):
    global chat_history
    recent_history = chat_history[-4:] if len(chat_history) >= 4 else chat_history
    api_history = format_history_for_api(recent_history)
    response = groq.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": f"You are a debate bot. Your name: llm_2. Goal/stance description: {goal_of_llm2}"},
            *api_history,
            {"role": "user", "content": input},
        ],
    )
    return response.choices[0].message.content

def stream_debate(user_goal, goal_of_llm1, goal_of_llm2, rounds=4, refined_goal="", assignment_json=None):
    global chat_history
    chat_history = [{"role": "user", "content": user_goal}]
    st.session_state["chat_history"] = chat_history
    
    # Display debate setup
    if assignment_json:
        st.markdown(f"""
        <div class="debate-card">
            <h3>🎯 Debate Topic</h3>
            <p><strong>{refined_goal}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="llm1-card">
                <h4>🤖 LLM 1 (Llama 3.1)</h4>
                <p><strong>Position:</strong> {goal_of_llm1}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="llm2-card">
                <h4>🧠 LLM 2 (Llama 4 Scout)</h4>
                <p><strong>Position:</strong> {goal_of_llm2}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create containers for real-time updates
    debate_container = st.container()
    
    with debate_container:
        for round_num in range(rounds):
            # Round header
            st.markdown(f"""
            <div class="round-header">
                🥊 Round {round_num + 1} of {rounds}
            </div>
            """, unsafe_allow_html=True)
            
            # Update progress
            progress = (round_num * 2) / (rounds * 2)
            progress_bar.progress(progress)
            status_text.text(f"Round {round_num + 1}: LLM 1 is thinking...")
            
            # LLM 1 turn
            llm_1_input = chat_history[-1]["content"]
            with st.spinner("LLM 1 is crafting response..."):
                llm_1_response = llm_1_call(llm_1_input, goal_of_llm1)
            chat_history.append({"role": "llm_1", "content": llm_1_response})
            
            st.markdown(f"""
            <div class="llm1-card">
                <h5>🤖 LLM 1 Response:</h5>
                <p>{llm_1_response}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Update progress
            progress = (round_num * 2 + 1) / (rounds * 2)
            progress_bar.progress(progress)
            status_text.text(f"Round {round_num + 1}: LLM 2 is thinking...")
            
            # LLM 2 turn
            llm_2_input = chat_history[-1]["content"]
            with st.spinner("LLM 2 is crafting response..."):
                llm_2_response = llm_2_call(llm_2_input, goal_of_llm2)
            chat_history.append({"role": "llm_2", "content": llm_2_response})
            
            st.markdown(f"""
            <div class="llm2-card">
                <h5>🧠 LLM 2 Response:</h5>
                <p>{llm_2_response}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add separator between rounds
            if round_num < rounds - 1:
                st.markdown("---")
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text(f"🎉 Debate completed! {rounds} rounds finished.")
    
    # Display completion message
    st.markdown(f"""
    <div class="debate-card">
        <h3>✅ Debate Concluded</h3>
        <p>The debate has concluded after {rounds * 2} exchanges between the AI participants.</p>
    </div>
    """, unsafe_allow_html=True)

def judge_gpt(chat_history):
    if not chat_history:
        return "No debate history available for judging."
    
    # Format the chat history for the judge
    formatted_history = ""
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_history += f"TOPIC: {msg['content']}\n\n"
        elif msg["role"] == "llm_1":
            formatted_history += f"LLM1: {msg['content']}\n\n"
        elif msg["role"] == "llm_2":
            formatted_history += f"LLM2: {msg['content']}\n\n"
    
    response = groq.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {"role": "system", "content": "You are the judge to decide the winner of a debate between 2 LLMs. Respond in this format strictly: WINNER: LLM1/LLM2 REASON: reason for choosing"},
            {"role": "user", "content": formatted_history},
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 AI Debate Arena</h1>
        <p>Watch two advanced AI models engage in structured intellectual discourse</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "debate_completed" not in st.session_state:
        st.session_state["debate_completed"] = False

    # Sidebar for controls
    with st.sidebar:
        st.markdown("""
        <div class="side-panel" style="color: #000;">
            <h3>⚙️ Debate Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        user_goal = st.text_area(
            "Enter your debate topic:",
            placeholder="e.g., Should artificial intelligence be regulated by governments?",
            height=100
        )
        
        rounds = st.slider(
            "Number of debate rounds:",
            min_value=1,
            max_value=4,
            value=3,
            help="Each round consists of one response from each AI"
        )
        
        st.markdown("""
        <div class="side-panel" style="color: #000;">
            <h4>🤖 AI Models</h4>
            <p><strong>LLM 1:</strong> Llama 3.1 (8B)</p>
            <p><strong>LLM 2:</strong> Llama 4 Scout (17B)</p>
            <p><strong>Judge:</strong> Gemma 2 (9B)</p>
        </div>
        """, unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Start Debate", type="primary", use_container_width=True) and user_goal:
            try:
                with st.spinner("Setting up the debate..."):
                    goal_of_llm1, goal_of_llm2, assignment_json = goal_refinement_and_assignment(user_goal)
                    refined_goal = assignment_json.get("refined_goal", user_goal)
                
                st.session_state["debate_completed"] = False
                stream_debate(user_goal, goal_of_llm1, goal_of_llm2, rounds, refined_goal, assignment_json)
                st.session_state["debate_completed"] = True
                
            except Exception as e:
                st.error(f"Error starting debate: {str(e)}")

    with col2:
        if st.button("🏆 Get Judge's Verdict", use_container_width=True):
            if st.session_state["chat_history"]:
                with st.spinner("Judge is analyzing the debate..."):
                    verdict = judge_gpt(st.session_state["chat_history"])
                
                st.markdown(f"""
                <div class="judge-card">
                    <h4>⚖️ Judge's Verdict</h4>
                    <p>{verdict}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No debate history available. Please start a debate first!")

    # Display stats if debate is completed
    if st.session_state["debate_completed"] and st.session_state["chat_history"]:
        st.markdown("---")
        st.markdown("### 📊 Debate Statistics")
        
        # Count messages
        llm1_count = len([msg for msg in st.session_state["chat_history"] if msg["role"] == "llm_1"])
        llm2_count = len([msg for msg in st.session_state["chat_history"] if msg["role"] == "llm_2"])
        total_exchanges = llm1_count + llm2_count
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-box" style="color: #000;">
            <h3>{llm1_count}</h3>
            <p>LLM 1 Responses</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-box" style="color: #000;">
            <h3>{llm2_count}</h3>
            <p>LLM 2 Responses</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-box" style="color: #000;">
            <h3>{total_exchanges}</h3>
            <p>Total Exchanges</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-box" style="color: #000;">
            <h3>{rounds}</h3>
            <p>Debate Rounds</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎭 Experience the future of AI discourse • Built with Streamlit & Groq</p>
    </div>
    """, unsafe_allow_html=True)