from app.services.preprocessing import tokenize

EMOJIS = {
    "joy": ["😊", "😄", "🎉"],
    "love": ["❤️", "🥰", "😍"],
    "sadness": ["😢", "💙", "🫂"],
    "anger": ["😤", "😡", "💢"],
    "fear": ["😟", "😰", "🫂"],
    "surprise": ["😮", "🤯", "✨"],
    "positive": ["😊", "✨", "👍"],
    "negative": ["😔", "💙", "🫂"],
    "neutral": ["💬", "📝", "🤔"],
}
KEYWORDS = {"birthday": "🎂", "coffee": "☕", "travel": "✈️", "exam": "📚", "win": "🏆"}


def recommend(text: str, sentiment: str, emotion: str, k: int = 3) -> list[str]:
    tokens = set(tokenize(text))
    candidates = [emoji for word, emoji in KEYWORDS.items() if word in tokens]
    candidates += EMOJIS.get(emotion, EMOJIS.get(sentiment, EMOJIS["neutral"]))
    return list(dict.fromkeys(candidates))[:k]
