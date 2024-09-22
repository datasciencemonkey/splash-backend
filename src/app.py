from dotenv import load_dotenv
import os
import dspy
import json
from datetime import datetime
import pytz
import replicate
from pydantic import BaseModel
from fastapi import FastAPI
import random
import uvicorn
from fastapi.responses import JSONResponse
from openai import OpenAI
# from databricks.sdk import WorkspaceClient
# import time
from langchain_community.tools import DuckDuckGoSearchResults


load_dotenv()
API_BASE = os.getenv("API_BASE")
API_KEY = os.getenv("API_KEY")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME")


def get_model():
    return dspy.Databricks(
        model="sg-external",
        model_type="chat",
        api_key=API_KEY,
        api_base=API_BASE,
        max_tokens=2000,
        temperature=round(0.7 + (random.randint(1, 100) / 10000), 4),
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
        post="Good morning, LinkedIn! ☀️ Kicking off a productive day with a team meeting, followed by an exciting client presentation this afternoon. Ready to showcase our latest innovations! #ProfessionalGrowth #ClientPresentation #InnovationInAction",
    ),
    dspy.Example(
        local_time="1:45 PM EDT",
        user_post="Looking forward to my presentation!",
        user_role="presenter",
        agenda=agenda,
        social_media_site="LinkedIn",
        rationale="The post should be professional yet engaging, suitable for Instagram. We'll focus on the upcoming presentation, incorporating the user's role as a presenter. The time suggests it's just before the presentation, so we'll emphasize preparation and excitement. We'll use relevant hashtags to increase visibility and engagement.",
        post="Pre-presentation butterflies! 🦋 Just 15 minutes until I take the stage to share our latest innovations with our valued clients. Months of hard work have led to this moment. Excited to showcase what our amazing team has accomplished! 💼✨ #ProfessionalGrowth #PublicSpeaking #InnovationPresentation #ReadyToInspire",
    ),
]

# Create and compile the model
post_generator = EngagingSocialMediaPost()
post_generator.generator.demos = examples


def get_current_time():
    eastern = pytz.timezone("US/Eastern")
    current_time = datetime.now(eastern)
    return current_time.strftime("%I:%M %p %Z")


class SocialMediaPostRequest(BaseModel):
    user_post: str
    user_role: str
    social_media_site: str


class SocialMediaPostResponse(BaseModel):
    post: str
    rationale: str


class ImgGenRequest(BaseModel):
    user_post: str
    negative_prompt: str = (
        """Do not include any text in the image or specific references to events.
        Do not include any words or any reference to databricks world tour"""
    )


class ImgPromptResponse(BaseModel):
    extracted_topics: str
    flux_prompt: str


class ImgGenSignature(dspy.Signature):
    user_post = dspy.InputField(desc="the social media post the user wants to make")
    negative_prompt = dspy.InputField(
        desc="include this statement to the prompt to avoid generating incorrect images"
    )
    extracted_topics = dspy.OutputField(
        desc="the extracted topics from the user's post relevant to Data and AI. must be specific to databricks. include mosaic ai if user is interested in AI"
    )
    flux_prompt = dspy.OutputField(
        desc="the prompt to be used for generating the image worthy of sharing in social media using flux1 image generation models.Focus on the ambience."
    )


class SocialMediaProcessor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt_generator = dspy.ChainOfThought(ImgGenSignature)

    def forward(self, user_post, negative_prompt):
        result = self.prompt_generator(
            user_post=user_post, negative_prompt=negative_prompt
        )
        return result


class ImgPromptRequest(BaseModel):
    img_prompt: str


class UserTopicsRequest(BaseModel):
    topics: str


def construct_messages_from_search(topics):
    search_query = f"""databricks blogs and videos related to the following databricks topics {topics}."""

    search = DuckDuckGoSearchResults()
    search_results = search.invoke(search_query)
    print(search_results)
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {
            "role": "user",
            "content": f"""Extract all the links from this text, produce a JSON array. 
            Only output this JSON and nothing else: {search_results}""",
        },
    ]
    return messages


app = FastAPI()


@app.post("/generate-social-media-post")
async def generate_social_media_post(
    request: SocialMediaPostRequest,
) -> SocialMediaPostResponse:
    current_time = get_current_time()
    with dspy.settings.context(lm=get_model()):
        response = post_generator(
            local_time=current_time,
            user_post=request.user_post,
            user_role=request.user_role,
            agenda=agenda,
            social_media_site=request.social_media_site,
        )
    return SocialMediaPostResponse(
        post=response.post.split("\n")[0], rationale=response.rationale
    )


@app.post("/generate-image-prompt-n-get-topics")
async def generate_image_prompt_n_get_topics(
    request: ImgGenRequest,
) -> ImgPromptResponse:
    processor = SocialMediaProcessor()
    with dspy.settings.context(lm=get_model()):
        response = processor(
            user_post=request.user_post, negative_prompt=request.negative_prompt
        )
    return ImgPromptResponse(
        extracted_topics=response.extracted_topics, flux_prompt=response.flux_prompt
    )


@app.post("/generate-image")
async def generate_image(request: ImgPromptRequest):
    output = replicate.run(
        IMAGE_MODEL_NAME,
        input={"prompt": request.img_prompt},
    )
    return JSONResponse(content={"image_url": output})


@app.post("/get-links-from-topics")
async def get_links_from_topics(request: UserTopicsRequest):
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
    )
    response = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=construct_messages_from_search(request.topics),
        max_tokens=2000,
        temperature=0.1,
    )
    json_response = response.choices[0].message.content
    result = json.loads(
        json_response[json_response.find("[") : json_response.find("]") + 1].strip("\n")
    )
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)