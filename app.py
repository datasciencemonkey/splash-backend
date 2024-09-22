from dotenv import load_dotenv
import os
import dspy
import json
from datetime import datetime
import pytz
from pydantic import BaseModel
from fastapi import FastAPI
import random

load_dotenv()

API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")

def get_model():
    return dspy.Databricks(
        model="sg-external",
        model_type="chat",
        api_key=API_KEY,
        api_base=API_BASE,
        max_tokens=2000,
        temperature=round(0.7+(random.randint(1,100)/10000),4)
    )

dspy.configure(lm=get_model())

with open("agenda.json", "r") as file:
    agenda = json.load(file)

agenda = json.dumps(agenda)

class SocialMediaPostGenerator(dspy.Signature):
    """You are a social media assistant who generates an engaging social media post with hashtags about the user's experience at the Databricks Data & AI World Tour 2024 (DAIWT) in Atlanta. This is a tech conference about Data and AI. Call out sessions where the local time overlaps with the user's post and assume that the user is one of the sessions if overlap happens between local time and session time. Each session is atleast 40 minutes. If there are multiple sessions that overlap, always choose AI sessions. If there are no AI sessions, choose based on current session.Give all posts a very positive spin to help it go viral. Ongoing sessions take priority, unless the user specifically calls out other sessions or topics"""

    local_time = dspy.InputField()
    user_post = dspy.InputField()
    user_role = dspy.InputField(
        desc="User's role in the conference. Either attendee or organizer or presenter"
    )
    agenda = dspy.InputField()
    social_media_site = dspy.InputField()
    rationale = dspy.OutputField(desc="Reasoning behind the post content and hashtags")
    current_session = dspy.OutputField(
        desc="Current session that the user could be in. If there isn't a match, return None"
    )
    post = dspy.OutputField(
        desc="Engaging social media post with hashtags. Pay specific attention to any current session the user is in.This is the ongoing session."
    )


class EngagingSocialMediaPost(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(SocialMediaPostGenerator)

    def forward(self, local_time, user_post, user_role, agenda, social_media_site):
        return self.generator(
            local_time=local_time,
            user_post=user_post,
            user_role=user_role,
            agenda=agenda,
            social_media_site=social_media_site,
        )

examples = [
    dspy.Example(
        local_time="09:30 AM EDT",
        user_post="Excited for my presentation today!",
        user_role="presenter",
        agenda=agenda,
        social_media_site="LinkedIn",
        rationale="The post should be professional and optimistic, highlighting the upcoming presentation. We'll use hashtags related to professional growth and presentations. The time suggests it's morning, so we can incorporate that.",
        post="Good morning, LinkedIn! ☀️ Kicking off a productive day with a team meeting, followed by an exciting client presentation this afternoon. Ready to showcase our latest innovations! #ProfessionalGrowth #ClientPresentation #InnovationInAction"
    ),
    dspy.Example(
        local_time="1:45 PM EDT",
        user_post="Looking forward to my presentation!",
        user_role="presenter",
        agenda=agenda,
        social_media_site="LinkedIn",
    rationale="The post should be professional yet engaging, suitable for Instagram. We'll focus on the upcoming presentation, incorporating the user's role as a presenter. The time suggests it's just before the presentation, so we'll emphasize preparation and excitement. We'll use relevant hashtags to increase visibility and engagement.",
    post="Pre-presentation butterflies! 🦋 Just 15 minutes until I take the stage to share our latest innovations with our valued clients. Months of hard work have led to this moment. Excited to showcase what our amazing team has accomplished! 💼✨ #ProfessionalGrowth #PublicSpeaking #InnovationPresentation #ReadyToInspire")
]

# Create and compile the model
post_generator = EngagingSocialMediaPost()
post_generator.generator.demos = examples

def get_current_time():
    eastern = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern)
    return current_time.strftime('%I:%M %p %Z')


class SocialMediaPostRequest(BaseModel):
    user_post: str
    user_role: str
    social_media_site: str

class SocialMediaPostResponse(BaseModel):
    post: str
    rationale: str


app = FastAPI()

@app.post("/generate-social-media-post")
async def generate_social_media_post(request: SocialMediaPostRequest) -> SocialMediaPostResponse:
    current_time = get_current_time()
    with dspy.settings.context(lm=get_model()):
        response = post_generator(
            local_time=current_time,
            user_post=request.user_post,
            user_role=request.user_role,
            agenda=agenda,
            social_media_site=request.social_media_site
        )
    return SocialMediaPostResponse(
        post=response.post.split('\n')[0],
        rationale=response.rationale
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)