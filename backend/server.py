from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
from enum import Enum
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
db_name = os.environ.get('DB_NAME', 'wine_tracker')
client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Enums
class WineType(str, Enum):
    RED = "Red"
    WHITE = "White"
    ROSE = "Rosé"
    SPARKLING = "Sparkling"
    ICE_WINE = "Ice Wine"

# Pydantic Models
class Wine(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: WineType
    cost: float
    country: str = ""
    fridge: str = ""
    pre_rating: Optional[int] = Field(None, ge=1, le=5)
    pairing: str = ""
    gift_from: str = ""
    used: bool = False
    consumption_rating: Optional[int] = Field(None, ge=1, le=5)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class WineCreate(BaseModel):
    name: str
    type: WineType
    cost: float
    country: str = ""
    fridge: str = ""
    pre_rating: Optional[int] = Field(None, ge=1, le=5)
    pairing: str = ""
    gift_from: str = ""

class WineUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[WineType] = None
    cost: Optional[float] = None
    country: Optional[str] = None
    fridge: Optional[str] = None
    pre_rating: Optional[int] = Field(None, ge=1, le=5)
    pairing: Optional[str] = None
    gift_from: Optional[str] = None

class WineUse(BaseModel):
    consumption_rating: int = Field(..., ge=1, le=5)

class WineSummary(BaseModel):
    total_value: float
    average_cost: float
    total_wines: int
    unused_wines: int
    used_wines: int

class WinePairingRequest(BaseModel):
    food_description: str
    cooking_method: Optional[str] = None
    occasion: Optional[str] = None
    preferences: Optional[str] = None

class WinePairingResponse(BaseModel):
    recommendations: str
    inventory_matches: List[Wine] = []
    session_id: str

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Wine Fridge Tracker API"}

@api_router.post("/wines", response_model=Wine)
async def create_wine(wine_data: WineCreate):
    """Add a new wine to the inventory"""
    wine_dict = wine_data.dict()
    wine = Wine(**wine_dict)
    
    try:
        await db.wines.insert_one(wine.dict())
        return wine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create wine: {str(e)}")

@api_router.get("/wines", response_model=List[Wine])
async def get_wines(pairing: Optional[str] = None):
    """Get all wines with optional pairing filter"""
    try:
        query = {}
        if pairing:
            query["pairing"] = {"$regex": pairing, "$options": "i"}  # Case-insensitive search
        
        wines_cursor = db.wines.find(query)
        wines = await wines_cursor.to_list(1000)
        return [Wine(**wine) for wine in wines]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch wines: {str(e)}")

@api_router.get("/wines/summary", response_model=WineSummary)
async def get_wine_summary():
    """Get inventory summary statistics"""
    try:
        # Get all wines
        wines_cursor = db.wines.find({})
        wines = await wines_cursor.to_list(1000)
        
        if not wines:
            return WineSummary(
                total_value=0.0,
                average_cost=0.0,
                total_wines=0,
                unused_wines=0,
                used_wines=0
            )
        
        total_wines = len(wines)
        unused_wines = [wine for wine in wines if not wine.get("used", False)]
        used_wines_count = total_wines - len(unused_wines)
        
        # Calculate total value (only unused wines)
        total_value = sum(wine.get("cost", 0) for wine in unused_wines)
        
        # Calculate average cost (only unused wines)
        average_cost = total_value / len(unused_wines) if unused_wines else 0.0
        
        return WineSummary(
            total_value=round(total_value, 2),
            average_cost=round(average_cost, 2),
            total_wines=total_wines,
            unused_wines=len(unused_wines),
            used_wines=used_wines_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@api_router.get("/wines/{wine_id}", response_model=Wine)
async def get_wine(wine_id: str):
    """Get a specific wine by ID"""
    try:
        wine_doc = await db.wines.find_one({"id": wine_id})
        if not wine_doc:
            raise HTTPException(status_code=404, detail="Wine not found")
        return Wine(**wine_doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch wine: {str(e)}")

@api_router.put("/wines/{wine_id}", response_model=Wine)
async def update_wine(wine_id: str, wine_update: WineUpdate):
    """Update wine details"""
    try:
        # Find the wine
        wine_doc = await db.wines.find_one({"id": wine_id})
        if not wine_doc:
            raise HTTPException(status_code=404, detail="Wine not found")
        
        # Prepare update data (only include non-None values)
        update_data = {k: v for k, v in wine_update.dict().items() if v is not None}
        update_data["updated_at"] = datetime.utcnow()
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No data provided for update")
        
        # Update the wine
        await db.wines.update_one(
            {"id": wine_id}, 
            {"$set": update_data}
        )
        
        # Return updated wine
        updated_wine_doc = await db.wines.find_one({"id": wine_id})
        return Wine(**updated_wine_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update wine: {str(e)}")

@api_router.put("/wines/{wine_id}/use", response_model=Wine)
async def use_wine(wine_id: str, wine_use: WineUse):
    """Mark a wine as used and add consumption rating"""
    try:
        # Find the wine
        wine_doc = await db.wines.find_one({"id": wine_id})
        if not wine_doc:
            raise HTTPException(status_code=404, detail="Wine not found")
        
        if wine_doc.get("used", False):
            raise HTTPException(status_code=400, detail="Wine already used")
        
        # Update the wine
        update_data = {
            "used": True,
            "consumption_rating": wine_use.consumption_rating,
            "updated_at": datetime.utcnow()
        }
        
        await db.wines.update_one(
            {"id": wine_id}, 
            {"$set": update_data}
        )
        
        # Return updated wine
        updated_wine_doc = await db.wines.find_one({"id": wine_id})
        return Wine(**updated_wine_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to use wine: {str(e)}")

@api_router.put("/wines/{wine_id}/reactivate", response_model=Wine)
async def reactivate_wine(wine_id: str):
    """Reactivate a used wine (mark as unused)"""
    try:
        # Find the wine
        wine_doc = await db.wines.find_one({"id": wine_id})
        if not wine_doc:
            raise HTTPException(status_code=404, detail="Wine not found")
        
        if not wine_doc.get("used", False):
            raise HTTPException(status_code=400, detail="Wine is not used")
        
        # Update the wine
        update_data = {
            "used": False,
            "consumption_rating": None,
            "updated_at": datetime.utcnow()
        }
        
        await db.wines.update_one(
            {"id": wine_id}, 
            {"$set": update_data}
        )
        
        # Return updated wine
        updated_wine_doc = await db.wines.find_one({"id": wine_id})
        return Wine(**updated_wine_doc)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reactivate wine: {str(e)}")

@api_router.delete("/wines/{wine_id}")
async def delete_wine(wine_id: str):
    """Delete a wine from inventory"""
    try:
        # Find the wine
        wine_doc = await db.wines.find_one({"id": wine_id})
        if not wine_doc:
            raise HTTPException(status_code=404, detail="Wine not found")
        
        # Delete the wine
        await db.wines.delete_one({"id": wine_id})
        
        return {"message": "Wine deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete wine: {str(e)}")

@api_router.post("/wines/ai-pairing", response_model=WinePairingResponse)
async def get_ai_wine_pairing(request: WinePairingRequest):
    """Get AI-powered wine pairing recommendations"""
    try:
        # Check if OpenAI API key is configured
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            raise HTTPException(
                status_code=503, 
                detail="AI wine pairing service is not configured. Please add your OpenAI API key to the .env file."
            )
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create AI chat instance
        chat = LlmChat(
            api_key=openai_api_key,
            session_id=session_id,
            system_message="""You are an expert sommelier and wine pairing specialist with decades of experience. 
            Your expertise includes knowledge of wine regions, varietals, vintages, and the art of food and wine pairing.
            
            When recommending wines:
            1. Consider the primary flavors, cooking methods, and seasonings of the dish
            2. Think about complementary vs. contrasting flavor profiles
            3. Consider the weight and intensity of both food and wine
            4. Suggest specific wine types, regions, and even grape varietals when appropriate
            5. Explain WHY each pairing works (the reasoning behind your recommendation)
            6. Give alternative options for different price ranges when possible
            7. Keep your response engaging and educational but concise
            
            Format your response as a friendly but knowledgeable sommelier would speak to a wine enthusiast."""
        ).with_model("openai", "gpt-4o")
        
        # Build the prompt
        prompt_parts = [f"I'm looking for wine pairing suggestions for: {request.food_description}"]
        
        if request.cooking_method:
            prompt_parts.append(f"Cooking method: {request.cooking_method}")
        
        if request.occasion:
            prompt_parts.append(f"Occasion: {request.occasion}")
        
        if request.preferences:
            prompt_parts.append(f"Additional preferences: {request.preferences}")
        
        prompt_parts.append("\nPlease provide 2-3 specific wine recommendations with explanations of why they pair well.")
        
        prompt = "\n".join(prompt_parts)
        
        # Get AI recommendation
        user_message = UserMessage(text=prompt)
        ai_response = await chat.send_message(user_message)
        
        # Get user's current wine inventory
        wines_cursor = db.wines.find({"used": False})  # Only available wines
        available_wines = await wines_cursor.to_list(1000)
        wines_list = [Wine(**wine) for wine in available_wines]
        
        # Find potential matches in user's inventory
        inventory_matches = []
        if wines_list:
            # Simple matching based on wine types mentioned in the AI response
            wine_types_in_response = ai_response.lower()
            
            for wine in wines_list:
                wine_type_lower = wine.type.lower()
                wine_name_lower = wine.name.lower()
                
                # Check if wine type or name is mentioned in the response
                if (wine_type_lower in wine_types_in_response or 
                    any(keyword in wine_types_in_response for keyword in wine_name_lower.split())):
                    inventory_matches.append(wine)
        
        return WinePairingResponse(
            recommendations=ai_response,
            inventory_matches=inventory_matches,
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI wine pairing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get wine pairing recommendations: {str(e)}")

# Original status check endpoints (keeping for compatibility)
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
  
