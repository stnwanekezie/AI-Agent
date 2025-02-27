# %%
import os
import logging
from openai import OpenAI
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from gmail import meeting_scheduler, meeting_notifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

model = "gpt-4o"
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION_ID"), 
    project=os.getenv("OPENAI_PROJECT_ID")
)

# %%
class EventExtraction(BaseModel):
    description: str = Field(..., description="Raw description of the event.")
    is_calendar_event: bool = Field(..., description="Whether the description is a calendar event.")
    confidence_score: float = Field(..., description="Confidence score of the prediction between 0 and 1.")

class EventDetails(BaseModel):
    name: str = Field(..., description="Name of the event.")
    date: str = Field(..., description="Date and time of the event. Use ISO 8601 to format this value.")
    duration_minutes: int = Field(..., description="Expected duration of the event in minutes.")
    participants: list[str] = Field(..., description="List of participants names of the event.")
    participant_emails: list[str] = Field(..., description=(
            "List of participants emails."
            "The format should be always '<last_name><first_name>@gmail.com' if not explicitly given."
        ) # '<first letter of first name>.<last name>
    )
    location: Optional[str] = Field(None, description="Location of the event.")


class EventConfirmation(BaseModel):
    confirmation_message: str = Field(..., description="Natural language confirmation message for the event.")
    calendar_link: Optional[str] = Field(None, description="Generated calendar link to the calendar event - if applicable.")

# %%
def extract_event_info(user_input: str) -> EventExtraction:

    logger.info('Starting event extraction analysis...')
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f'Today is {today.strftime(r"%A, %B %d, %Y")}.'
    system_prompt = f"{date_context} Analyze if the text describes a calendar event."
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        response_format=EventExtraction
    )

    result = completion.choices[0].message.parsed
    
    logger.info(f"Event extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score}")
    
    return result

def parse_event_details(description: str) -> EventDetails:
    """Second LLM call to extract specific event details"""
    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime(r'%A, %B %d, %Y')}."

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{date_context} Extract detailed event information."
                    "When dates reference 'next Tuesday' or similar relative dates, use this current date as reference."
                ),
            },
            {"role": "user", "content": description},
        ],
        response_format=EventDetails,
    )
    result = completion.choices[0].message.parsed
    logger.info(
        f"Parsed event details - Name: {result.name}, Date: {result.date}, Duration: {result.duration_minutes}min"
    )
    logger.debug(f"Participants: {', '.join(result.participants)}")

    return result

def generate_confirmation(event_details: EventDetails, new_time: datetime, meeting_link: str) -> EventConfirmation:
    """Third LLM call to generate a confirmation message"""
    logger.info("Generating confirmation message")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate a natural confirmation message for the event."
                    f"The scheduled meeting time is {new_time.strftime(r'%A, %B %d, %Y')}"
                    f"and meeting location is {event_details.location or meeting_link}." 
                    "Sign off with your name; Stanley."
                ),
            },
            {"role": "user", "content": str(event_details.model_dump())},
        ],
        response_format=EventConfirmation,
    )
    result = completion.choices[0].message.parsed
    logger.info("Confirmation message generated successfully")
    
    return result

def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """Main function implementing the prompt chain with gate check"""
    logger.info("Processing calendar request")
    logger.debug(f"Raw input: {user_input}")

    # First LLM call: Extract basic info
    initial_extraction = extract_event_info(user_input)

    # Gate check: Verify if it's a calendar event with sufficient confidence
    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confidence_score < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_calendar_event: {initial_extraction.is_calendar_event}, confidence: {initial_extraction.confidence_score:.2f}"
        )
        return None

    logger.info("Gate check passed, proceeding with event processing")

    # Second LLM call: Get detailed event information
    event_details = parse_event_details(initial_extraction.description)
    new_time, meeting_id, meeting_link = meeting_scheduler(
        event_details.name, datetime.fromisoformat(event_details.date), 
        event_details.duration_minutes, event_details.participant_emails
    )

    logger.info(f"Meeting scheduled successfully - ID: {meeting_id}, Link: {meeting_link}")
    
    # Third LLM call: Generate confirmation
    confirmation = generate_confirmation(event_details, new_time, meeting_link)
    meeting_notifier(event_details.participant_emails, event_details.name, confirmation.confirmation_message)

    logger.info("Calendar request processing completed successfully")
    
    return confirmation

# %%
user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Iyama Gilbert to discuss the project roadmap."

result = process_calendar_request(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")

