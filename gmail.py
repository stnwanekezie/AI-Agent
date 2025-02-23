# %%
import base64
import time
import io
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import gspread
# from fpdf import FPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

root = Path(__file__).parent.resolve()
# 🔒 Secure OAuth Credentials
with open(root / "credentials.json", "r") as f:
    CREDENTIALS_FILE = json.load(f)
# CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")

# ✅ API SCOPES (Least Privilege)
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/spreadsheets"
]

# %%

# ✅ API Rate Limit Settings
REQUEST_DELAY = 1  # Wait 1 second between API requests

# 🔑 Authenticate Google API Services
def authenticate_google():
    logger.info("Authenticating Google API Services...")

    creds = None
    token = root / "token.json"
    if token.exists():
        creds = Credentials.from_authorized_user_file(token, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=65183)
        with open(token, "w") as token:
            token.write(creds.to_json())
    return (
        build("calendar", "v3", credentials=creds),
        build("gmail", "v1", credentials=creds),
        gspread.service_account_from_dict(CREDENTIALS_FILE)
    )

# %%
# 🔍 Check for existing meetings in Google Calendar
def check_existing_meeting(start_time):
    # event_title, 
    # Returns True if a meeting exists at the same time
    logger.info("Checking for existing meetings in Google Calendar...")

    calendar_service, _, _ = authenticate_google()
    events_result = calendar_service.events().list(
        calendarId="primary",
        # q=event_title,
        timeMin=start_time.isoformat() + 'Z', # 'Z' indicates UTC time,
        maxResults=5,
        singleEvents=True,
        orderBy="startTime"
    ).execute()
    return len(events_result.get("items", [])) > 0  

# 🕒 Find the next available time slot
def find_next_available_slot(start_time, meeting_duration=30):
    logger.info("Finding the next available time slot...")

    calendar_service, _, _ = authenticate_google()
    time_slot = start_time
    while True:
        time_slot += timedelta(minutes=30)  # Check every 30 minutes
        events = calendar_service.events().list(
            calendarId="primary",
            timeMin=time_slot.isoformat() + 'Z',
            timeMax=(time_slot + timedelta(minutes=meeting_duration)).isoformat() + 'Z',
            singleEvents=True
        ).execute().get("items", [])
        if not events:
            return time_slot  # Found available slot

# 📅 Create a Meeting in Google Calendar
def create_meeting(event_title, start_time, end_time, attendees, time_zone=None, recurring=None):
    logger.info("Creating a meeting in Google Calendar...")
    calendar_service, _, _ = authenticate_google()    
    event_body = {
        "summary": event_title,
        "start": {"dateTime": start_time.isoformat(), "timeZone": time_zone},
        "end": {"dateTime": end_time.isoformat(), "timeZone": time_zone},
        "attendees": [{"email": email.strip()} for email in attendees],
        "conferenceData": {"createRequest": {"requestId": f"meet-{int(datetime.now().timestamp())}"}},
        "visibility": "private"
    }
    
    # 🔄 Add Recurrence
    if recurring:
        recurrence_map = {"daily": "DAILY", "weekly": "WEEKLY", "monthly": "MONTHLY"}
        if recurring.lower() in recurrence_map:
            event_body["recurrence"] = [f"RRULE:FREQ={recurrence_map[recurring.lower()]}"]

    event = calendar_service.events().insert(calendarId="primary", body=event_body, conferenceDataVersion=1).execute()
    time.sleep(REQUEST_DELAY)
    return event["id"], event.get("hangoutLink", "No Google Meet Link")

# 📧 Send Email with PDF Attachment
def send_email(email_list, subject, body, pdf_attachment=None):
    logger.info(f"Sending email to {', '.join(email_list)}...")
    _, gmail_service, _ = authenticate_google()
    message = MIMEMultipart()
    message["to"] = ", ".join(email_list)
    message["subject"] = subject
    message.attach(MIMEText(body, "plain"))

    if pdf_attachment:
        attachment = MIMEApplication(pdf_attachment.read(), _subtype="pdf")
        attachment.add_header("Content-Disposition", "attachment", filename="Meeting_Details.pdf")
        message.attach(attachment)

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    gmail_service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
    time.sleep(REQUEST_DELAY)

# %%
def meeting_scheduler(event_title, start_time, meeting_duration, attendees, time_zone=None, recurring=None):
    conflicts = check_existing_meeting(start_time)
    if conflicts:
        new_time = find_next_available_slot(start_time, meeting_duration)
        print(f"⚠️ Conflict detected! Rescheduling to {new_time.strftime(r'%A, %B %d, %Y')}")
        start_time = new_time

    end_time = start_time + timedelta(minutes=meeting_duration)
    
    if not time_zone:
        time_zone = "Europe/London"

    meeting_id, meeting_link = create_meeting(event_title, start_time, end_time, attendees, time_zone=time_zone, recurring=recurring)
    
    return new_time, meeting_id, meeting_link 

def meeting_notifier(email_list, subject, body, pdf_attachment=None):
    try:
        send_email(email_list, subject, body, pdf_attachment)
        print(f"✉️ Email successfully sent to {email_list}")
    except Exception as e:
        print(f"❌ Error sending email: {e}")
    