from pydantic import BaseModel, Field

class VideoClassification(BaseModel):
    video_id: str = Field(..., description="The video's ID")
    game_title: str = Field(..., description="The identified game title or 'META'")
    genre_category: str = Field(..., description="The category number as a string if known, else 'META'")
    is_gaming: bool = Field(..., description="True if about gaming, else false")
