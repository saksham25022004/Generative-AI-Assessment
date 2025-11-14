# Task 2 — Weather Agent 

## Story

Our internal marketing team wants a friendly assistant that can answer questions about the weather for any city worldwide.

They want to use this as a starting point for an AI-powered chatbot that can later handle more complex user queries.

Your goal is to build a **Weather Agent** — a small FastAPI-based service that:

1. Understands a user's question in natural language (e.g., *"What's the weather in Paris today?"*)
2. Extracts the city name from the text using an **LLM (OpenAI)**
3. Fetches the real-time weather data from **OpenWeatherMap API**
4. Returns a **clear, natural-sounding summary** as a JSON response

---

## What's Provided

Inside your `agent_task/` folder, you'll find:

| File | Description |
|------|-------------|
| `agent_weather.py` | The main FastAPI app with `/ask` endpoint and placeholders (`# TODO`) for you to implement improvements |
| `sample_prompts.txt` | Example user prompts to test your API |
| `README_RESULTS.md` | Write a short summary of what you changed and why |

You'll need your **OpenAI API key** and **OpenWeatherMap API key** (both can be added to `.env`).

---

## Step-by-Step Mission

### Understand the Flow

Open `agent_weather.py` — the file contains:

- An `/ask` POST endpoint that accepts JSON:
  ```json
  { "prompt": "What's the weather in Paris today?" }
  ```

- **Functions:**
  - `extract_city()` — identifies the city from the text (currently simple)
  - `get_weather()` — fetches live weather data via REST API
  - `ask()` — orchestrates the two steps and returns a formatted result

You'll see several `# TODO` comments in the file — these are your improvement zones.

---

### Improve City Extraction

Right now, `extract_city()` uses an OpenAI prompt plus a simple regex fallback. It's functional but can fail for tricky sentences like:

- *"Do I need an umbrella in London?"*
- *"Is it hot in Delhi right now?"*
- *"Weather forecast for San Francisco tomorrow?"*

**Your task:**

- Make the LLM prompt more reliable
- If the LLM fails or returns "NONE," use a regex fallback
- Optionally, add logic to strip unwanted words or punctuation

**Example improvement idea:**

*"Extract only the city name from this question. If multiple cities are mentioned, pick the first one."*

---

### Handle API Errors Gracefully

The function `get_weather()` calls OpenWeatherMap. Sometimes the API can fail (invalid key, city not found, etc.).

**Your task:**

Add error handling for:
- Invalid city names
- HTTP errors (non-200 responses)
- Missing fields in JSON
- **Optional:** Add retry logic with exponential backoff for failed calls

If the weather API call fails, return a friendly JSON error like:

```json
{ "error": "Could not fetch weather for Paris." }
```

---

### Make the Response User-Friendly

The `/ask` endpoint should return something clean, like:

```json
{
  "answer": "It's currently 27°C and clear sky in Paris. Humidity: 65%. Wind: 4.2 m/s.",
  "raw": {
    "city": "Paris",
    "temp": 27,
    "desc": "clear sky",
    "humidity": 65,
    "wind": 4.2
  }
}
```

**Your task:**

- Ensure the `answer` string is readable and concise
- Handle pluralization (e.g., "1 meter per second" vs "3 meters per second")
- Optionally round the numbers to 1 decimal place

---

### Test the API

Start your service:

```bash
cd agent_task
uvicorn agent_weather:app --reload --port 8002
```

Then open your browser:

```
http://127.0.0.1:8002/docs
```

Use the `/ask` endpoint and try different queries:

- *"What's the weather in New York?"*
- *"Do I need a jacket in Berlin?"*
- *"Is it raining in Mumbai today?"*

---

## Deliverables

By the end of this task, you should have:

- A working `/ask` endpoint that handles various prompts gracefully
- Clear, natural responses formatted for users
- Updated `agent_weather.py` with improvements
- Notes in `README_RESULTS.md` describing your changes

---

## Hints

- Use `units=metric` in the API URL to get °C
- The OpenWeatherMap free plan is enough for this task
- You can reuse your LLM from other tasks — GPT-4o-mini works perfectly
- Keep the service lightweight and fast — no need for databases

---