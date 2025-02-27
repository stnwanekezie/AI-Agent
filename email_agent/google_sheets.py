# # %%
# # ✅ Google Sheets Config
# GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "your_google_sheet_id")
# GOOGLE_SHEET_RANGE = "Meetings!A2:F"

# # 📄 Generate PDF Attachment
# def generate_meeting_pdf(event_title, start_time, meeting_link):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, "Meeting Details", ln=True, align="C")
#     pdf.ln(10)
#     pdf.cell(200, 10, f"Title: {event_title}", ln=True)
#     pdf.cell(200, 10, f"Date & Time: {start_time.strftime('%Y-%m-%d %H:%M %Z')}", ln=True)
#     pdf.cell(200, 10, f"Meeting Link: {meeting_link}", ln=True)
    
#     pdf_output = io.BytesIO()
#     pdf.output(pdf_output, "F")
#     pdf_output.seek(0)
#     return pdf_output

# # 📊 Read Meetings from Google Sheets
# def read_meetings_from_sheets():
#     _, _, sheets_client = authenticate_google()
#     sheet = sheets_client.open_by_key(GOOGLE_SHEET_ID).worksheet("Meetings")
#     return [row for row in sheet.get(GOOGLE_SHEET_RANGE) if len(row) >= 5]

# # 🚀 Book Meetings and Prevent Conflicts
# def book_meetings_from_sheets():
#     meetings = read_meetings_from_sheets()

#     for i, meeting in enumerate(meetings, start=2):
#         event_title, date, start_time, attendees_str, recurring = meeting[:5]
#         attendees = attendees_str.split(",")
#         start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
#         end_dt = start_dt + timedelta(hours=1)

#         # Detect conflicts
#         if check_existing_meeting(start_dt): # event_title, 
#             new_time = find_next_available_slot(start_dt)
#             print(f"⚠️ Conflict detected! Rescheduling '{event_title}' to {new_time}")
#             start_dt, end_dt = new_time, new_time + timedelta(hours=1)

#         # Book Meeting
#         event_id, meeting_link = create_meeting(event_title, start_dt, end_dt, attendees, recurring)

#         # Generate PDF
#         pdf_attachment = generate_meeting_pdf(event_title, start_dt, meeting_link)

#         # Send Email
#         email_subject = f"Meeting Scheduled: {event_title}"
#         email_body = f"Meeting Link: {meeting_link}\nDate & Time: {start_dt.strftime('%Y-%m-%d %H:%M')}"
#         for attendee in attendees:
#             send_email(attendee.strip(), email_subject, email_body, pdf_attachment)

# # 🔥 Run the Script
# book_meetings_from_sheets()