from openai import OpenAI
import json
from stqdm import stqdm
import prompts
import llm_clients

def generate_json_resume(cv_text, api_key, model="gpt-4o"):
    """Generate a JSON resume from a CV text"""
    sections = []
    client = OpenAI(api_key=api_key)

    for prompt in stqdm(
        [
            prompts.BASICS_PROMPT,
            prompts.EDUCATION_PROMPT,
            prompts.AWARDS_PROMPT,
            prompts.PROJECTS_PROMPT,
            prompts.SKILLS_PROMPT,
            prompts.WORK_PROMPT,
        ],
        desc="This may take a while...",
    ):
        filled_prompt = prompt.replace(prompts.CV_TEXT_PLACEHOLDER, cv_text)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": filled_prompt},
            ],
        )

        try:
            answer = response.choices[0].message.content
            answer = json.loads(answer)

            if prompt == prompts.BASICS_PROMPT and "basics" not in answer:
                answer = {"basics": answer}  # common mistake GPT makes

            sections.append(answer)
        except Exception as e:
            print(e)

    final_json = {}
    for section in sections:
        final_json.update(section)

    return final_json


def tailor_resume(cv_text, api_key, model="gpt-4o"):
    filled_prompt = prompts.TAILORING_PROMPT.replace("<CV_TEXT>", cv_text)
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_TAILORING},
                {"role": "user", "content": filled_prompt},
            ],
        )

        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(e)
        print("Failed to tailor resume.")
        return cv_text
