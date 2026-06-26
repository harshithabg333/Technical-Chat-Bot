#Import the dependencies
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel

#Load Environmental Varaibles
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGO_DB_URI")

#Connect to MongoDB
client = MongoClient(mongo_uri)

db = client["techchat"]


collection = db["users"]

app = FastAPI()

class ChatRequest(BaseModel):
    user_id :str
    question:str

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers=["*"],
    allow_credentials= True    
)


#collection.delete_many({})
print("Success!")
#prompt for our chat bot #Setting up langchain prompt
prompt = ChatPromptTemplate.from_messages([
    
    ("system","You are a Tech Chat Bot"),
    ("placeholder","{history}"),
    ("user","{question}")

])

llm = ChatGroq(api_key = groq_api_key, model = "openai/gpt-oss-20b") #Continuation from this 

#pipe It links the prompt structure directly into the LLM. When you invoke this chain, 
# it pushes the data through the prompt, 
# formats it perfectly, and feeds it straight to the model.
chain = prompt | llm


#Helper Function: Fetching Chat History
#It loops through those messages and structures them into tuples of (role, message)—like ("user", "Hi"),
# ("ai", "Hello!")—which LangChain expects. It then returns this array list.

def get_history(user_id):
    # 1. Fetches raw chat logs from MongoDB database
    chats = collection.find({"user_id":user_id}).sort("timestamp",1)
    #THIS IS YOUR ARRAY LIST! It starts completely empty: []
    history =[]
    
    # 3. Loops through each chat message found in the database   
    for chat in chats :
        # 4. Appends a structured tuple (role, message) into your list
        history.append((chat["role"],chat["message"]))   
     # 5. Returns the populated array list to be used by LangChain   
    return history 
#Endpoints (The API Routes)Defines a simple HTTP GET endpoint at the root URL (/). If you open http://localhost:8000/ in a web browser,
# it will simply return a JSON welcome message.
@app.get("/")
def home():
    return {"message" : "Welcome to the Tech bot"}

#Creates an HTTP POST route at /chat.
# It expects to receive a data payload structured like the ChatRequest model defined earlier
@app.post("/chat")
def chat(request:ChatRequest):
   history = get_history(request.user_id) #Calls the helper function to fetch all previous back-and-forth messages for this unique user from the database.
   
   response = chain.invoke({"history":history,"question":request.question}) 
  #What it does: Fires the formatted data to your LangChain chain. It inputs the chat history and the current question, 
  # waits for the model to think, and stores the AI's response object into response.
   collection.insert_one({ #Logs the user's current question into the MongoDB database

        "user_id" : request.user_id, #"The roles system, user, and assistant 
        "role":"user", #user,system
        "message" : request.question,
        "timestamp" : datetime.utcnow()
    })
   collection.insert_one({ #Logs the AI's newly generated answer
        "user_id" : request.user_id,
        "role" : "ai", #"The roles system, user, and assistant /ai
        "message" : response.content,
        "timestamp" : datetime.utcnow()
    })   

   return {"response" : response.content}  
#Sends the AI's response text back as a JSON







