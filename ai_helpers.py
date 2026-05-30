import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


def get_api_key():
    """Get OpenAI API key - tries st.secrets first, falls back to env var."""
    try:
        import streamlit as st
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")


def analyze_roster_image(uploaded_file):
    client = OpenAI(api_key=get_api_key())

    image_bytes = uploaded_file.read()

    base64_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
                Analyze the work roster screenshot.

                Return ONLY valid JSON.

                Format:

                {
                    "month": "",
                    "year": 2026,
                    "scheduled_days": [],
                    "summary": ""
                }

                Only extract scheduled days.
                """
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this roster."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )

    ai_response = response.choices[0].message.content

    cleaned_response = (
        ai_response
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    data = json.loads(cleaned_response)

    data["scheduled_days"] = sorted(
        [int(day) for day in data["scheduled_days"]]
    )

    return data, ai_response
