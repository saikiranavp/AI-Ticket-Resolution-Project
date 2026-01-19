import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.vectorstores import FAISS

load_dotenv()
hf_token = st.secrets["HF_TOKEN"]


llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
       
    )
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


problems = [
    
    {
        "content": "Incorrect Credentials: Users mistype usernames or passwords, leading to repeated failed login attempts.",
        "category": "Login Problems",
        "tags": ["login", "credentials", "authentication"]
    },
    {
        "content": "Forgotten Passwords: Password reset flows are confusing or insecure, causing frustration.",
        "category": "Login Problems",
        "tags": ["login", "password_reset", "security"]
    },
    {
        "content": "Account Lockouts: Multiple failed attempts lock users out unnecessarily.",
        "category": "Login Problems",
        "tags": ["login", "lockout", "rate_limit"]
    },
    {
        "content": "Two-Factor Authentication Issues: OTP codes not received or mismatched authenticator tokens.",
        "category": "Login Problems",
        "tags": ["login", "2fa", "otp"]
    },
    {
        "content": "Device/Browser Compatibility: Login pages fail on certain browsers or mobile devices.",
        "category": "Login Problems",
        "tags": ["login", "compatibility", "browser", "mobile"]
    },

   
    {
        "content": "Duplicate Accounts: Users unintentionally create multiple accounts due to unclear signup flows.",
        "category": "Signup Problems",
        "tags": ["signup", "duplicate", "ux"]
    },
    {
        "content": "Email Verification Failures: Verification emails not delivered or marked as spam.",
        "category": "Signup Problems",
        "tags": ["signup", "email_verification", "deliverability"]
    },
    {
        "content": "Weak Password Restrictions: Users struggle with password policies that reject simple passwords.",
        "category": "Signup Problems",
        "tags": ["signup", "password_policy", "security"]
    },
    {
        "content": "Form Validation Errors: Poorly designed error messages confuse users when fields are incomplete or invalid.",
        "category": "Signup Problems",
        "tags": ["signup", "validation", "ux"]
    },
    {
        "content": "Third-Party Integration Failures: Signup via Google/Slack fails due to API or permission issues.",
        "category": "Signup Problems",
        "tags": ["signup", "oauth", "integration"]
    },

   
    {
        "content": "Delayed Responses: Tickets remain unresolved due to backlog or poor prioritization.",
        "category": "Ticket Resolution Problems",
        "tags": ["tickets", "sla", "prioritization"]
    },
    {
        "content": "Inconsistent Knowledge Base: Agents provide conflicting answers because of outdated or incomplete documentation.",
        "category": "Ticket Resolution Problems",
        "tags": ["tickets", "knowledge_base", "documentation"]
    },
    {
        "content": "Poor Categorization: Tickets are misclassified, slowing down routing to the right team.",
        "category": "Ticket Resolution Problems",
        "tags": ["tickets", "categorization", "routing"]
    },
    {
        "content": "Limited Context: Agents lack full history of customer interactions, leading to repetitive questions.",
        "category": "Ticket Resolution Problems",
        "tags": ["tickets", "context", "history"]
    },
    {
        "content": "Escalation Bottlenecks: Tickets requiring higher-level support get stuck without clear escalation paths.",
        "category": "Ticket Resolution Problems",
        "tags": ["tickets", "escalation", "workflow"]
    },

    
    {
        "content": "Automated Tagging Errors: NLP misclassifies ticket intent, causing irrelevant recommendations.",
        "category": "Smart Support System Challenges",
        "tags": ["system", "tagging", "nlp"]
    },
    {
        "content": "Recommendation Accuracy Issues: Suggested articles don’t match the actual problem context.",
        "category": "Smart Support System Challenges",
        "tags": ["system", "recommendation", "accuracy"]
    },
    {
        "content": "Feedback Loop Gaps: Lack of structured feedback prevents the system from learning and improving.",
        "category": "Smart Support System Challenges",
        "tags": ["system", "feedback", "learning"]
    },
    {
        "content": "Integration Failures: Sync issues with external tools like Google Sheets or Slack.",
        "category": "Smart Support System Challenges",
        "tags": ["system", "integration", "sheets", "slack"]
    },
    {
        "content": "Analytics Blind Spots: Missing metrics on article usage and ticket resolution trends.",
        "category": "Smart Support System Challenges",
        "tags": ["system", "analytics", "metrics"]
    },
]

texts = [p["content"] for p in problems]
metas = [{"category": p["category"], "tags": p["tags"]} for p in problems]


db = FAISS.from_texts(texts, embeddings, metadatas=metas)
retriever = db.as_retriever(search_kwargs={"k": 5})


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support assistant. Answer ONLY using the provided context. "
           "If the context does not contain relevant information, say 'No relevant information found in knowledge base.'"),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{question}\n\nContext:\n{context}")
])
chain = prompt | llm


stores = {}
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in stores:
        stores[session_id] = InMemoryChatMessageHistory()
    return stores[session_id]


def get_context(question: str):
    return retriever.invoke(question)

def ask_with_history(question: str, session_id: str = "default_session") -> str:
    history = get_history(session_id)
    docs = get_context(question)
    if not docs:
        return "No relevant information found in knowledge base." 
    context = "\n".join([d.page_content for d in docs])
    history.add_message(HumanMessage(content=question))
    response = chain.invoke({
        "context": context,
        "messages": history.messages,
        "question": question
    })
    history.add_message(AIMessage(content=response.content))
    return response.content


st.title("Smart Support & Ticket Resolution Assistant")

session_id = "default_session"
query = st.text_area("Enter ticket text or a question", height=120)

if st.button("Get Recommendation"):
    
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    answer = ask_with_history(query, session_id=session_id)
    st.write(answer)



