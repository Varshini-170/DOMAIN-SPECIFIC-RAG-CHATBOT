"""Create two small, safe-to-share PDFs for a first demo of the chatbot."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


OUTPUT_DIR = Path(__file__).resolve().parent / "documents"


def build_pdf(filename: str, title: str, pages: list[list[str]]) -> None:
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(OUTPUT_DIR / filename), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm, title=title,
    )
    story = []
    for page_index, paragraphs in enumerate(pages):
        story.append(Paragraph(title, styles["Title"]))
        story.append(Spacer(1, 0.3 * cm))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodyText"]))
            story.append(Spacer(1, 0.2 * cm))
        if page_index < len(pages) - 1:
            story.append(PageBreak())
    document.build(story)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_pdf(
        "BrightPath_Employee_Handbook.pdf",
        "BrightPath Employee Handbook - Demo",
        [
            [
                "Welcome to BrightPath Learning Services. This fictional handbook is a sample document for a student RAG project; it is not a real company policy.",
                "Working hours are 9:30 AM to 6:30 PM, Monday through Friday. Team members should notify their manager as early as possible if they expect to arrive late.",
                "Attendance is calculated monthly as the number of approved working days attended divided by scheduled working days. Approved leave and public holidays are excluded from scheduled working days.",
            ],
            [
                "Employees may work remotely up to two days per week with manager approval. Remote work must be agreed before the start of the workday.",
                "The company CEO for this fictional training example is Anita Rao. The Head of Operations is Dev Kumar.",
                "For an equipment problem, create a ticket with the IT help desk and include the device asset number.",
            ],
        ],
    )
    build_pdf(
        "BrightPath_Leave_Policy.pdf",
        "BrightPath Leave Policy - Demo",
        [
            [
                "This fictional leave policy is supplied only for demonstrating retrieval-augmented generation.",
                "Full-time employees receive 18 paid annual leave days per calendar year. Leave accrues at 1.5 days at the end of each completed month.",
                "Submit planned leave through the employee portal at least five working days in advance. The manager should approve or decline the request within two working days.",
            ],
            [
                "Employees receive 10 sick leave days per calendar year. For an absence of three or more consecutive working days, a medical certificate may be requested.",
                "Unused annual leave may carry over up to five days into the next calendar year. Carried-over days expire on 31 March.",
                "Compassionate leave provides up to three paid working days for the death of an immediate family member. Contact Human Resources for support.",
            ],
        ],
    )
    print(f"Created sample PDFs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
