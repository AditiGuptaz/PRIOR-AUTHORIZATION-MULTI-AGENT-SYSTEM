"""
Runs the full Prior Authorization pipeline for one request:
Extractor -> Policy Checker -> (Writer <-> Reviewer loop) -> done.

The Writer/Reviewer loop is the one piece CrewAI doesn't hand you for
free the way LangGraph's conditional edges did. So this is a plain
Python loop: write a letter, have it reviewed, and if rejected, write
a new one that addresses the feedback -- up to a maximum number of
attempts, so it can't loop forever.
"""

import json

import nest_asyncio
nest_asyncio.apply()

from crewai import Crew
from agents import (
    extractor_agent, extraction_task,
    policy_checker_agent, policy_check_task,
    writer_agent, reviewer_agent,
    make_writing_task, make_review_task,
)


def run_prior_authorization(request: dict, payer_policy_text: str, max_attempts: int = 3):
    # Step 1 & 2 only need to run once -- extraction and policy checking
    # don't change no matter how many times the letter gets rewritten.
    intake_crew = Crew(
        agents=[extractor_agent, policy_checker_agent],
        tasks=[extraction_task, policy_check_task],
    )
    intake_crew.kickoff(inputs={
        "raw_clinical_note": request["raw_clinical_note"],
        "payer_policy_text": payer_policy_text,
    })

    feedback = None
    current_writing_task = None
    review_output = None

    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Writer/Reviewer attempt {attempt} ---")

        current_writing_task = make_writing_task(feedback)
        current_review_task = make_review_task(current_writing_task)

        draft_crew = Crew(
            agents=[writer_agent, reviewer_agent],
            tasks=[current_writing_task, current_review_task],
        )
        draft_crew.kickoff()

        review_output = json.loads(str(current_review_task.output))

        if review_output["approved"]:
            print("Approved by reviewer.")
            return str(current_writing_task.output), review_output

        feedback = review_output["feedback"]
        print(f"Sent back with feedback: {feedback}")

    print("Max attempts reached without approval.")
    return str(current_writing_task.output), review_output


if __name__ == "__main__":
    from pypdf import PdfReader

    reader = PdfReader("payer_policy.pdf")
    payer_policy_text = "\n".join(p.extract_text() for p in reader.pages)

    with open("prior_auth_requests.json") as f:
        data = json.load(f)

    request = data["requests"][1]  # PA002 -- the one that should trigger the loop

    letter, decision = run_prior_authorization(request, payer_policy_text)

    print("\n=== FINAL LETTER ===")
    print(letter)
    print("\n=== FINAL DECISION ===")
    print(decision)
