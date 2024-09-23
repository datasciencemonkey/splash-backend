# Social Media Post Generator

This project is designed to generate engaging social media posts, images, and recommendations based on user input. It leverages various APIs and libraries to create content that can help users go viral on social media platforms.

## Overview

The pipeline in `src/social-post.py` enables the following steps:

1. **Load Environment Variables:**
   - The script starts by loading environment variables from a `.env` file using `load_dotenv()`. These variables include `API_BASE`, `API_KEY`, `ENDPOINT_NAME`, and `IMAGE_MODEL_NAME`.

2. **Configure the Model:**
   - The `get_model` function configures a Databricks model using the `dspy` library. This model is used for generating social media posts and image prompts.

3. **Load Agenda:**
   - The script loads an agenda from a JSON file (`agenda.json`) and converts it to a JSON string.

4. **Define Social Media Post Generator:**
   - A `SocialMediaPostGenerator` class is defined using `dspy.Signature`. This class specifies the inputs and outputs for generating a social media post.

5. **Define Engaging Social Media Post Module:**
   - An `EngagingSocialMediaPost` class is defined using `dspy.Module`. This class uses the `SocialMediaPostGenerator` to generate social media posts.

6. **Define Examples:**
   - A list of examples is defined to demonstrate how the social media post generator works. These examples include sample inputs and expected outputs.

7. **Create and Compile the Model:**
   - An instance of `EngagingSocialMediaPost` is created and compiled with the examples.

8. **Define Helper Functions:**
   - `get_current_time`: Returns the current time in the US/Eastern timezone.
   - `construct_messages_from_search`: Constructs messages for searching Databricks blogs and videos using DuckDuckGo.
   - `generate_social_media_post`: Generates a social media post using the configured model.
   - `generate_image_prompt_n_get_topics`: Generates an image prompt and extracts topics from a user's post.
   - `generate_image_from_prompt`: Generates an image from a prompt using the Replicate API.
   - `get_links_from_topics`: Gets links related to specific topics using the OpenAI API.
   - `get_recommendations`: Gets recommendations based on extracted topics.

9. **Define Update Function:**
   - The `update` function orchestrates the generation of social media posts, images, and recommendations. It takes user inputs and generates the required outputs.

10. **Create Gradio Interface:**
    - A Gradio interface is created to allow users to interact with the pipeline. The interface includes input fields for the social media post, user type, social media app, and checkboxes for generating images and recommendations.
    - The `update` function is connected to the Gradio interface, and the outputs are displayed in textboxes and image components.

11. **Launch the Gradio Interface:**
    - The Gradio interface is launched with specific settings, including hiding the footer and logo.

## Summary of the Pipeline

1. **User Input:**
   - The user provides input through the Gradio interface, including the text for the social media post, user type, social media app, and options for generating images and recommendations.

2. **Generate Social Media Post:**
   - The `generate_social_media_post` function generates a social media post based on the user's input and the current time.

3. **Generate Image and Recommendations:**
   - If the user opts to generate an image, the `generate_image_prompt_n_get_topics` function generates an image prompt and extracts topics from the user's post.
   - The `generate_image_from_prompt` function generates an image based on the prompt.
   - If the user opts to generate recommendations, the `get_recommendations` function fetches related links based on the extracted topics.

4. **Display Output:**
   - The generated social media post, image, and recommendations are displayed in the Gradio interface.

This pipeline enables users to generate engaging social media posts, images, and recommendations based on their input, making it easier to create content for social media platforms.
