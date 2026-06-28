# ========================================================
# 1. DEPENDENCY IMPORTS
# ========================================================

# Imports the built-in system module to interact with environmental operating system variables
import os

# Imports the load_dotenv function to read configuration keys from a local hidden `.env` file
from dotenv import load_dotenv

# Imports the ChatGroq client class from LangChain to handle inference requests with high-speed LLM models
from langchain_groq import ChatGroq

# Imports the ChatPromptTemplate engine to structure conversation components cleanly into roles
from langchain_core.prompts import ChatPromptTemplate

# Imports MongoClient to establish persistent connection streams to your live MongoDB cluster instance
from pymongo import MongoClient

# Imports the standard datetime module to create high-resolution timeline stamps for records
from datetime import datetime

# Imports the FastAPI framework class to manage routing behaviors and build our network API layer
from fastapi import FastAPI

# Imports CORSMiddleware to securely clear cross-origin request processing rules inside web browsers
from fastapi.middleware.cors import CORSMiddleware 

# Imports BaseModel from Pydantic to build automated data structure validation guardrails for requests
from pydantic import BaseModel


# ========================================================
# 2. APPLICATION INITIALIZATION & STORAGE ARCHITECTURE
# ========================================================

# Locates and parses the local `.env` configuration file to load hidden security strings into memory
load_dotenv()

# Grabs your private API token string from the operating system's active runtime environment setup
groq_api_key = os.getenv("GROQ_API_KEY")

# Retrieves your full cloud connection string path containing server routing logs for MongoDB
mongo_uri = os.getenv("MONGO_DB_URI")

# Initializes the persistent network driver client container targeting your cloud database instance
client = MongoClient(mongo_uri)

# Selects or dynamically sets up a dedicated backend operational database named "techchat"
db = client["techchat"]

# Selects or dynamically setups up the target document writing table collection named "users"
collection = db["users"]

# Instantiates the primary core FastAPI web service server application instance window wrapper
app = FastAPI()


# ========================================================
# 3. DATA CONTRACT VALIDATION & CORS SECURITY WIRING
# ========================================================

# Defines a customized schema template to enforce incoming transactional payload rules using Pydantic
class ChatRequest(BaseModel):
    # Dictates that the input payload structure must explicitly contain a text-string field named 'user_id'
    user_id :str
    # Dictates that the input payload structure must explicitly contain a text-string field named 'question'
    question:str

# Attaches security verification parameters to the running web application instance window setup
app.add_middleware(
    # Specifies that the injected layer explicitly uses the standard cross-origin resource toolset
    CORSMiddleware,
    # Opens communications to accept payloads originating from any remote web domain address placeholder (*)
    allow_origins = ["*"],
    # Authorizes standard web application interaction methods like POST, GET, PUT, and DELETE over network routes
    allow_methods = ["*"],
    # Grants clearance for incoming request headers containing complex structural parameters or auth metadata
    allow_headers=["*"],
    # Activates verification cookies or session authentication tracking parameters over structural pipeline lanes
    allow_credentials= True    
)

# Prints a verification feedback string straight into your server terminal to confirm script execution success
print("Success!")


# ========================================================
# 4. LANGCHAIN EXPRESSION COMPOSITION (LCEL PIPELINE)
# ========================================================

# Compiles systematic messaging layers together into a structured prompt schema tracking template container
prompt = ChatPromptTemplate.from_messages([
    # Feeds structural role guidance text down into the underlying core system behaviors layer
    ("system","You are a Tech Chat Bot"),
    # Opens a dynamic placeholder array index slot to inject historical context logs into the pipeline
    ("placeholder","{history}"),
    # Maps the user's latest message query directly into the final interaction template lane slot
    ("user","{question}")
])

# Connects to the Groq processing engine instance using your secure key and maps to a target model variant
llm = ChatGroq(api_key = groq_api_key, model = "openai/gpt-oss-20b") 

# Merges the prompt template and LLM object using the pipe (|) operator to create an executable LangChain Expression Language (LCEL) chain
chain = prompt | llm


# ========================================================
# 5. CONVERSATION HISTORICAL CONTEXT HELPERS
# ========================================================

# Declares a functional tracking helper engine designed to locate contextual background logs using a unique user key
def get_history(user_id):
    # Queries the MongoDB database table collection for logs matching the user ID and sorts them oldest to newest
    chats = collection.find({"user_id":user_id}).sort("timestamp",1)
    # Creates an empty array list tracking structure designed to parse documents step by step
    history =[]
    
    # Executes an evaluation loop traversal over every raw document record pulled from the database cluster
    for chat in chats :
        # Extracts raw document keys and appends them as a paired text tuple structure (role, message) into your list
        history.append((chat["role"],chat["message"]))   
        
    # Hands back the cleanly structured timeline list container back to the invoking pipeline block
    return history 


# ========================================================
# 6. HTTP API NETWORK ROUTE ENDPOINTS
# ========================================================

# Binds an asynchronous HTTP GET listener pattern directly onto the primary base network context route (/)
@app.get("/")
# Defines the operational block execution loop logic for the home server location indicator
def home():
    # Dispatches a clean text-based welcoming dictionary object structure directly back across the network layout
    return {"message" : "Welcome to the Tech bot"}

# Binds an asynchronous HTTP POST handler process directly onto the primary interaction routing lane (/chat)
@app.post("/chat")
# Defines the operational transactional pipeline block requiring data validation passing matching the ChatRequest contract
def chat(request:ChatRequest):
   # Triggers your extraction utility function to fetch previous historical dialogue pairs using the user's ID string
   history = get_history(request.user_id) 
   
   # Pushes the history list and new question string down the LCEL pipeline chain to calculate response parameters
   response = chain.invoke({"history":history,"question":request.question}) 
   
   # Triggers a transaction write command to append the user's question tracking document details to the database collection
   collection.insert_one({ 
        # Stores the unique identification identifier matching the client workspace trace line
        "user_id" : request.user_id, 
        # Injects a role verification flag identifying the text payload source block as a user event action
        "role":"user", 
        # Writes the raw question string query payload directly into the storage table row layout slot
        "message" : request.question,
        # Attaches an accurate high-resolution Universal Time Coordinated timestamp mark to the storage event write
        "timestamp" : datetime.utcnow()
    })
   
   # Triggers a secondary consecutive write command to save the AI's response content back inside the database cluster
   collection.insert_one({ 
        # Tracks the transaction under the exact same matching client target identification footprint line
        "user_id" : request.user_id,
        # Injects a role tracking flag identifying the string payload as an automated model execution layer output
        "role" : "ai", 
        # Writes the raw generated response text content string directly into the message storage field property
        "message" : response.content,
        # Captures and logs the exact execution timeline stamp detailing when the response content landed back safely
        "timestamp" : datetime.utcnow()
    })   

   # Streams a network response dictionary tracking structure carrying the calculated answer payload safely back to the client device
   return {"response" : response.content}  

