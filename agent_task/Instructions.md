# Task 2 — Weather Agent

Build a FastAPI weather agent that answers natural language questions using OpenAI and OpenWeatherMap.

---

## What You Need to Implement

### 1. `fetch_weather(city: str)`

Fetch real-time weather data from OpenWeatherMap.

**API Details:**

- **Endpoint:** `http://api.openweathermap.org/data/2.5/weather`
- **Params:** `q={city}&appid={WEATHER_API_KEY}&units=metric`

**Requirements:**

- Use `requests.get()` with a timeout
- Check `response.status_code`
- Extract relevant fields from JSON:
  - `temp`
  - `description`
  - `humidity`
  - `wind speed`
- Handle errors gracefully:
  - Invalid city names
  - HTTP errors (non-200 responses)
  - Network issues
- Return a **weather dict** on success, `None` on failure

---

### 2. `generate_response(weather_data: dict, original_prompt: str)`

Use OpenAI to create a natural language answer.

**Requirements:**

- Use OpenAI's **function/tool calling** to extract the city from the user prompt
- Define `fetch_weather` as a tool for OpenAI to call
- Let OpenAI decide **when** to call the weather tool
- Generate a conversational response based on the weather data
- Answer the user's specific question, for example:
  - "Do I need a jacket?"
  - "What's the temperature?"

**Tool Definition Example:**

```python
pythontools = [{
    "type": "function",
    "function": {
        "name": "fetch_weather",
        "description": "Get current weather for a city",
        "parameters": {
            # your JSON schema here
        }
    }
}]
```

---

### 3. `ask_weather(query: WeatherQuery)`

Orchestrate the complete workflow.

**Flow:**

1. Call OpenAI with the user prompt and the weather tool definition
2. If OpenAI calls the tool, execute `fetch_weather(city)`
3. Send the weather data back to OpenAI for the **final response**
4. Return a JSON object like:

   ```json
   { "answer": "natural language response" }
   ```

**Error Handling:**

- If anything fails, return a JSON error like:

  ```json
  { "error": "error message" }
  ```

---

## Testing

Run the server:

```bash
cd agent_task
uvicorn agent_weather:app --reload --port 8002
```

Open the interactive docs:

```
http://127.0.0.1:8002/docs
```

Try example prompts with the `/ask` endpoint:

- "What's the weather in Paris?"
- "Do I need an umbrella in London?"
- "Is it hot in Mumbai right now?"

---

## Expected Response Format

On success:

```json
{
  "answer": "It's currently 27°C with clear skies in Paris. Light wind at 4.2 m/s. Perfect weather!"
}
```

On error:

```json
{
  "error": "Could not fetch weather for InvalidCity"
}
```

---

Use this guide to implement or refactor `agent_weather.py` so that it uses OpenAI tool calling and OpenWeatherMap together in a clean, robust way.
