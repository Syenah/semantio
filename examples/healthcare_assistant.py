# Set the Groq API key (either via environment variable or explicitly)
import os
os.environ["GROQ_API_KEY"] = "YOUR-API-KEY"  # Set the API key here

# Initialize the Agent
from semantio.agent import Agent
healthcare_research_assistant = Agent(
    name="Healthcare Agent",
    description="Extract and structure medical information from the provided text into a JSON format used in healthcare",
    instructions=[
        "Always use medical terminology while creating json",
        "Extract and structure medical information from the provided text into a JSON format used in healthcare",
        "Don't stringify the JSON",
    ],
    model="Groq",
    user_name="Researcher",
    markdown=True,
)
patient_text = """
Patient Complaints of High grade fever, chest pain, radiating towards right shoulder. Sweating,
patient seams to have high grade fever ,  patient is allergic to pollution , diagnosis high grade fever , plan of care comeback after 2 days , instructions take rest and drink lot of water  Palpitation since 5 days.
Advice investigation: CBC, LFT, Chest X ray, Abdomen Ultrasound
Medication: Diclofenac 325mg twice a day for 5 days, Amoxiclave 625mg once a day for 5 days, Azithromycin 500mg Once a day
Ibuprofen SOS, Paracetamol sos, Pentoprazol before breakfast  , follow up after 2 days
"""
# Test the Agent
healthcare_research_assistant.print_response(patient_text)