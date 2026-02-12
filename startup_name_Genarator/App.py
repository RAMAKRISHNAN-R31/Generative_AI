import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Startup Name Generator",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 Startup Name Generator")
st.markdown("**Created by Ramakrishnan Raman**")

# ---------------- Sidebar Inputs ----------------

categories = [
    "AI",
    "Health Tech",
    "FinTech",
    "EdTech",
    "E-commerce",
    "Gaming"
]

selected_category = st.sidebar.selectbox(
    "Choose your startup category:",
    categories
)

generate = st.sidebar.button("Generate 🚀")

# ---------------- LLM ----------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.9
)

# ---------------- Prompts ----------------
startup_name_prompt = PromptTemplate(
    input_variables=["domain"],
    template="""
Create ONLY ONE unique and catchy startup name for a {domain} startup.
Rules:
- Return only the name
- No explanations
- No extra text
"""
)

domain_prompt = PromptTemplate(
    input_variables=["startup_name"],
    template="""
Suggest ONLY ONE professional .com domain name for the startup "{startup_name}".
Rules:
- Return only the domain
- No explanations
- No extra text
"""
)

# ---------------- Sequential Logic ----------------
def sequential_chain(inputs):
    domain = inputs["domain"]

    startup_name = llm.invoke(
        startup_name_prompt.format(domain=domain)
    ).content.strip()

    domain_name = llm.invoke(
        domain_prompt.format(startup_name=startup_name)
    ).content.strip()

    return {
        "startup_name": startup_name,
        "domain_name": domain_name
    }

chain = RunnableLambda(sequential_chain)

# ---------------- Run App ----------------
if generate:

    # Priority: typed domain > category
    final_domain = selected_category

    with st.spinner("Generating startup name and domain..."):
        response = chain.invoke({"domain": final_domain})

    st.success("Done!")

    st.header(response["startup_name"])
    st.subheader("🌐 Domain Name")
    st.write(response["domain_name"])
