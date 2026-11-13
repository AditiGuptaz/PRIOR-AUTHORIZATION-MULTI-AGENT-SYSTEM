"""
CrewAI agents for the Prior Authorization multi-agent system.

Each Agent below is the equivalent of a system prompt — who this
worker is and what they're good at. Each Task is the equivalent of
a user prompt — the specific job, with real data filled into the
{blank} placeholders, same pattern as Day 11/12.
"""

import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew

llm = "gpt-4o-mini"


# ---------------------------------------------------------------------
# AGENT 1 — Extractor
# ---------------------------------------------------------------------

extractor_agent = Agent(
    role="Clinical Data Extractor",
    goal="Extract structured facts from a raw clinical note for prior authorization review",
    backstory=(
        "You are an experienced clinical data analyst who specializes in reading "
        "physician notes and pulling out exactly the facts needed for insurance "
        "review, without adding any interpretation beyond what is documented."
    ),
    llm=llm,
    verbose=True,
)

extraction_task = Task(
    description=(
        "Read the following clinical note and extract structured facts.\n\n"
        "CLINICAL NOTE:\n{raw_clinical_note}\n\n"
        "Return ONLY valid JSON with these exact fields:\n"
        "diagnosis (string), procedure_requested (string), "
        "conservative_therapy_weeks (number or null), "
        "red_flags_present (list of strings, empty list if none), "
        "exam_findings (list of strings), "
        "prior_mri_same_condition (true or false), "
        "months_since_prior_mri (number or null), "
        "documented_change_in_status (true, false, or null)."
    ),
    expected_output="A single JSON object with the exact fields listed above, and nothing else.",
    agent=extractor_agent,
)


# ---------------------------------------------------------------------
# AGENT 2 — Policy Checker
# ---------------------------------------------------------------------

policy_checker_agent = Agent(
    role="Payer Policy Reviewer",
    goal="Determine whether a prior authorization request meets each criterion in the payer's policy",
    backstory=(
        "You are an experienced utilization review specialist. You apply payer "
        "criteria exactly as written, including any documented exceptions, and "
        "you never approve or deny based on assumptions not supported by the "
        "extracted clinical facts."
    ),
    llm=llm,
    verbose=True,
)

policy_check_task = Task(
    description=(
        "Here is the payer's prior authorization policy:\n\n{payer_policy_text}\n\n"
        "Using the extracted clinical facts provided as context from the previous "
        "task, determine whether this request meets each of the 4 numbered "
        "criteria in the policy.\n\n"
        "For each criterion, decide: Met, Not Met, Insufficient Information, or — "
        "only for Criterion 2 — Waived (only if a red flag finding is documented). "
        "Criterion 4 should be Not Applicable if there was no prior MRI for the "
        "same condition.\n\n"
        "Return ONLY valid JSON with these fields: criterion_1, criterion_2, "
        "criterion_3, criterion_4, and overall_decision (Approved, Denied, or "
        "Needs More Information)."
    ),
    expected_output="A single JSON object with the 5 fields described above, and nothing else.",
    agent=policy_checker_agent,
    context=[extraction_task],
)


# ---------------------------------------------------------------------
# AGENT 3 — Writer
# ---------------------------------------------------------------------

writer_agent = Agent(
    role="Prior Authorization Writer",
    goal="Draft a clear, complete prior authorization letter based on the policy review",
    backstory=(
        "You are a prior authorization specialist who writes clear, well-justified "
        "letters to payers. You always reference the specific criteria that were "
        "met, and you are honest about any criteria that were not met or where "
        "information is missing."
    ),
    llm=llm,
    verbose=True,
)


def make_writing_task(feedback=None):
    """
    Builds a fresh Writing task each time it's called. If feedback from a
    previous Reviewer rejection is passed in, it gets added to the
    instructions — same fill-in-the-blank idea as everywhere else.
    """
    extra = f"\n\nIMPORTANT — address this feedback from the previous review:\n{feedback}" if feedback else ""
    return Task(
        description=(
            "Using the extracted clinical facts and the criteria results from the "
            "previous steps, draft a prior authorization letter.\n\n"
            "If any criterion is 'Not Met' or 'Insufficient Information', the letter "
            "must clearly state this and explain exactly what additional "
            "documentation or action is needed, rather than overstating the case."
            f"{extra}\n\n"
            "Return the letter as plain text, addressed to the payer's "
            "utilization review department."
        ),
        expected_output="A complete prior authorization letter as plain text.",
        agent=writer_agent,
        context=[extraction_task, policy_check_task],
    )


# ---------------------------------------------------------------------
# AGENT 4 — Reviewer
# ---------------------------------------------------------------------

reviewer_agent = Agent(
    role="Prior Authorization Reviewer",
    goal="Catch any letter that overstates the case or is missing required justification before it is sent",
    backstory=(
        "You are a senior reviewer who checks every prior authorization letter "
        "before it goes out. You compare the letter against the actual criteria "
        "results and reject any letter that claims approval is justified when a "
        "criterion was Not Met or Insufficient, or where supporting documentation "
        "is missing."
    ),
    llm=llm,
    verbose=True,
)


def make_review_task(writing_task):
    """
    Builds a fresh Review task tied to whichever writing_task it needs to
    check — this changes each loop attempt, so it can't be defined once
    up front the way the other tasks are.
    """
    return Task(
        description=(
            "Review the drafted letter against the actual criteria results from "
            "the policy check.\n\n"
            "If overall_decision was 'Approved' and every criterion is Met or "
            "Waived, approve the letter as is.\n\n"
            "If any criterion is 'Not Met' or 'Insufficient Information', the "
            "letter must NOT claim full approval. Check whether the letter "
            "honestly reflects this. If it does not, reject it and explain "
            "exactly what the Writer needs to fix.\n\n"
            "Return ONLY valid JSON with these fields: approved (true or false), "
            "feedback (string, empty string if approved)."
        ),
        expected_output="A single JSON object with the 2 fields described above, and nothing else.",
        agent=reviewer_agent,
        context=[extraction_task, policy_check_task, writing_task],
    )


if __name__ == "__main__":
    import json

    with open("prior_auth_requests.json") as f:
        data = json.load(f)

    one_request = data["requests"][0]  # PA001

    crew = Crew(agents=[extractor_agent], tasks=[extraction_task])
    result = crew.kickoff(inputs={"raw_clinical_note": one_request["raw_clinical_note"]})
    print(result)
