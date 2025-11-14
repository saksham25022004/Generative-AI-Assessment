"""
=========================================================
Task 2 — WEATHER AGENT (FastAPI + OpenAI + OpenWeatherMap)
=========================================================

This service takes a natural language question such as:
    "Will it rain in Bangalore tomorrow?"
    "What's the temperature in New York?"

It then:
    1) Extracts the city from the prompt
    2) Calls the OpenWeatherMap API
    3) Returns a natural language summary

---------------------------------------
TODO ITEMS
---------------------------------------
1. Improve city extraction logic 
   - Make LLM prompt more robust
   - Add safer regex fallback
   - Handle multi-word cities, typos, etc.

2. Improve weather API logic
   - Add retry with exponential backoff
   - Validate API response schema
   - Handle invalid city vs API errors gracefully

3. Improve response formatting
   - Support follow-up questions
   - Make summary more natural

4. Add logging & exception handling
   - Required for production-quality service
"""

from fastapi import FastAPI
from pydantic import BaseModel
import os
import re
import requests
from dotenv import load_dotenv
import openai

# ------------------------------------------------------
# Load environment variables
# ------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
openai.api_key = OPENAI_API_KEY

# ------------------------------------------------------
# FastAPI App
# ------------------------------------------------------
app = FastAPI(title="Weather Agent Service")

# ------------------------------------------------------
# Request Model
# ------------------------------------------------------
class Ask(BaseModel):
    prompt: str


# ------------------------------------------------------
# City Extraction
# ------------------------------------------------------
def extract_city(prompt_text: str) -> str | None:
    """
    Attempts to extract a city name from the user prompt.

    TODO for candidate:
    -------------------
    - Improve the LLM prompt (handle cases like "weather tomorrow in New Delhi")
    - Add message role:system for better control
    - Add regex fallback for multi-word cities
    - Add confidence scoring or a second-pass validation
    """

    # LLM-based extraction (Primary)
    try:
        llm_prompt = (
            "Extract ONLY the city name from the following message. "
            "If no city is mentioned, respond exactly with NONE.\n\n"
            f"Message: {prompt_text}"
        )

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract city names only."},
                {"role": "user", "content": llm_prompt},
            ]
        )

        city = response["choices"][0]["message"]["content"].strip()
        if city.upper() == "NONE":
            return None
        return city

    except Exception:
        pass  # If LLM fails, fallback to regex

    # Regex Fallback
    # TODO: Improve pattern to handle "weather at", "in the city of", etc.
    match = re.search(r"in ([A-Za-z ]+)", prompt_text, re.IGNORECASE)
    return match.group(1).strip() if match else None


# ------------------------------------------------------
# Weather Retrieval
# ------------------------------------------------------
def get_weather(city: str) -> dict | None:
    """
    Calls the OpenWeatherMap API and returns clean weather data.

    TODO for candidate:
    -------------------
    - Add retry logic (3 retries, exponential backoff)
    - Validate JSON keys before accessing
    - Handle "city not found" error (404)
    - Support future forecast API (bonus)
    """

    url = (
        f"http://api.openweathermap.org/data/2.5/weather?"
        f"q={city}&appid={WEATHER_API_KEY}&units=metric"
    )

    try:
        response = requests.get(url, timeout=8)

        if response.status_code != 200:
            return None

        data = response.json()

        # Basic validation
        if "main" not in data or "weather" not in data:
            return None

        return {
            "city": city,
            "temp": data["main"]["temp"],
            "desc": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind": data["wind"]["speed"],
        }

    except Exception:
        return None


# ------------------------------------------------------
# FastAPI Route
# ------------------------------------------------------
@app.post("/ask")
def ask_weather(request: Ask):
    """
    Returns a natural language weather summary.

    Candidate TODO:
    ----------------
    - Improve formatting (e.g., "It might rain later today")
    - Add support for follow-up questions in a session
    - Add richer weather details (pressure, cloud cover, sunrise, etc.)
    """

    # Extract city from user prompt
    city = extract_city(request.prompt)
    if not city:
        return {"error": "Could not determine the city from your question."}

    # Fetch weather info
    weather = get_weather(city)
    if not weather:
        return {"error": f"Unable to fetch weather for '{city}'."}

    # Generate a natural summary
    summary = (
        f"In {weather['city']}, it's currently {weather['temp']}°C with "
        f"{weather['desc']}. Humidity is {weather['humidity']}% and "
        f"wind speed is {weather['wind']} m/s."
    )

    return {
        "answer": summary,
        "raw": weather
    }
