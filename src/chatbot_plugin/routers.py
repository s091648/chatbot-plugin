"""FastAPI router for the chatbot plugin.

Mount this router into the scrape-and-analyze backend to enable chat endpoints.

Usage::

    from chatbot_plugin.routers import chat_router
    app.include_router(chat_router, prefix="/chat", tags=["chat"])
"""

from fastapi import APIRouter

chat_router = APIRouter()


@chat_router.post("/message")
async def send_message(message: str) -> dict:
    """Send a message to the chatbot and receive a response."""
    # TODO: implement chat logic
    return {"reply": f"Echo: {message}"}
