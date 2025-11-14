"""
Weather Agent Assessment
========================
Build a weather agent that answers natural language questions using:
- OpenAI for understanding user intent
- OpenWeatherMap API for weather data
- Use Tool calls and pass fetch_weather function as a tool
- FastAPI for the service

Example: "Will it rain in Bangalore tomorrow?" → Natural language weather summary
"""

from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import openai
import requests

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

app = FastAPI(title="Weather Agent")


class WeatherQuery(BaseModel):
    prompt: str


def fetch_weather(city: str) -> dict | None:
    """
    TODO: Call OpenWeatherMap API to get current weather.
    
    Hints:
    - API endpoint: http://api.openweathermap.org/data/2.5/weather
    - Parameters: q={city}, appid={key}, units=metric
    - Handle errors: invalid city, API failures, timeouts
    - Add retry logic for robustness
    
    Returns:
        Weather data dict or None on failure
    """
    # YOUR CODE HERE
    pass


def generate_response(weather_data: dict, original_prompt: str) -> str:
    """
    TODO: Use OpenAI to generate natural language weather summary.
    
    Hints:
    - Provide weather data as context
    - Answer the user's specific question
    - Make response conversational and helpful
    
    Returns:
        Natural language weather summary
    """
    # YOUR CODE HERE
    pass


@app.post("/ask")
def ask_weather(query: WeatherQuery):
    """
    TODO: Orchestrate the complete weather agent workflow.
    
    Steps:
    1. Extract city from prompt
    2. Fetch weather data
    3. Generate natural response
    4. Handle errors gracefully
    
    Returns:
        {"answer": str} or {"error": str}
    """
    # YOUR CODE HERE
    pass

if __name__ == "__main__":
    result = fetch_weather("New York")

    print("Result:", result)