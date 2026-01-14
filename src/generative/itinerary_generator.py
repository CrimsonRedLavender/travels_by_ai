import ollama

def generate_itinerary(destination, days, budget, interests):
    prompt = f"""
    Create a {days}-day travel itinerary for {destination}.
    Budget level: {budget}.
    Traveler interests: {interests}.

    Include:
    - Morning, afternoon, evening activities
    - Restaurants or food suggestions
    - Sightseeing highlights
    - Local tips

    Format the answer clearly by day.
    """

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
