import os
from fpdf import FPDF

# Create a PDF from the study techniques text file
def create_pdf_from_text(text_file, pdf_file):
    # Read the text file
    with open(text_file, 'r') as file:
        content = file.read()
    
    # Create PDF with wider margins
    pdf = FPDF()
    pdf.set_margins(20, 20, 20)  # Set wider margins (left, top, right)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)  # Use Helvetica instead of Arial with smaller font
    
    # Split content into lines and add to PDF
    lines = content.split('\n')
    for line in lines:
        if line.startswith('# '):
            # Main title
            pdf.set_font("Helvetica", 'B', 14)
            pdf.cell(170, 10, text=line[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
        elif line.startswith('## '):
            # Section title
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(170, 10, text=line[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
        elif line.startswith('### '):
            # Subsection title
            pdf.set_font("Helvetica", 'B', 11)
            pdf.cell(170, 10, text=line[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
        elif line.strip() == '':
            # Empty line
            pdf.ln(5)
        else:
            # Regular text - handle long lines safely
            try:
                # Use multi_cell with updated parameters
                pdf.multi_cell(170, 5, text=line)
            except Exception as e:
                print(f"Warning: Could not add line: {line[:30]}... - {str(e)}")
                # Try adding the line word by word if it fails
                words = line.split()
                safe_line = ""
                for word in words:
                    try:
                        pdf.multi_cell(170, 5, text=word)
                    except:
                        print(f"Warning: Skipping word: {word}")
    
    # Save the pdf
    pdf.output(pdf_file)
    print(f"Created PDF: {pdf_file}")

# Create a simple PDF directly without using text files
def create_simple_pdf(title, content, pdf_file):
    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    
    # Add title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(170, 10, text=title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Add content
    pdf.set_font("Helvetica", size=10)
    for paragraph in content:
        pdf.multi_cell(170, 5, text=paragraph)
        pdf.ln(5)
    
    # Save the pdf
    pdf.output(pdf_file)
    print(f"Created PDF: {pdf_file}")

if __name__ == "__main__":
    # Create documents directory if it doesn't exist
    os.makedirs('/home/ubuntu/educareer_guide/documents', exist_ok=True)
    
    # Create study techniques PDF
    study_techniques_content = [
        "Effective studying is a skill that can be learned and improved over time. This guide presents research-backed study techniques.",
        "Spaced repetition is a learning technique that involves reviewing material at increasing intervals. Instead of cramming all at once, space your study sessions over time.",
        "Active recall involves actively stimulating memory during the learning process. Instead of passively reading notes, test yourself on the material.",
        "The Pomodoro Technique is a time management method that uses a timer to break work into intervals, traditionally 25 minutes in length, separated by short breaks.",
        "Mind mapping is a visual organization technique that helps connect ideas and concepts.",
        "The Feynman Technique involves explaining concepts in simple terms as if teaching a child."
    ]
    create_simple_pdf("Study Techniques for Academic Success", study_techniques_content, 
                     '/home/ubuntu/educareer_guide/documents/study_techniques.pdf')
    
    # Create career guide PDF
    career_guide_content = [
        "A career path is a sequence of job positions that make up your career. Understanding potential career paths in your field can help you set realistic goals.",
        "Your resume is often your first impression with potential employers. A well-crafted resume can significantly increase your chances of landing interviews.",
        "Preparing for interviews involves researching the company, practicing common questions, and developing strategies to showcase your skills and experience.",
        "Professional networking is crucial for career development. A strong network can provide job leads, mentorship, and industry insights.",
        "Negotiating salary is an important skill that can significantly impact your earnings throughout your career.",
        "Ongoing learning and skill development are essential for long-term career success in today's rapidly changing job market."
    ]
    create_simple_pdf("Career Development Guide", career_guide_content,
                     '/home/ubuntu/educareer_guide/documents/career_guide.pdf')
    
    # Create financial aid guide PDF
    financial_aid_content = [
        "Financial aid comes in various forms, including grants, scholarships, loans, and work-study programs.",
        "The Free Application for Federal Student Aid (FAFSA) is the gateway to most federal and many state financial aid programs.",
        "Scholarships can significantly reduce your education costs without adding debt. A strategic approach to scholarship searching can yield better results.",
        "Winning scholarships requires more than just finding opportunities - you need to submit compelling applications.",
        "If you need to borrow for education, understanding how to manage student loans is essential for your financial future.",
        "Developing financial literacy skills while in school sets the foundation for lifelong financial health."
    ]
    create_simple_pdf("Financial Aid and Scholarship Guide", financial_aid_content,
                     '/home/ubuntu/educareer_guide/documents/financial_aid.pdf')
    
    # Create skills development guide PDF
    skills_development_content = [
        "The job market is constantly evolving, with certain skills becoming increasingly valuable across industries.",
        "Online learning platforms offer flexible, accessible ways to develop new skills.",
        "A structured learning plan increases your chances of successfully acquiring new skills.",
        "A portfolio showcases your skills and accomplishments to potential employers or clients.",
        "Industry certifications can validate your skills and enhance your credentials.",
        "Working with mentors or coaches can accelerate skill development and career growth."
    ]
    create_simple_pdf("Skills Development and Training Guide", skills_development_content,
                     '/home/ubuntu/educareer_guide/documents/skills_development.pdf')
    
    print("Sample PDF documents created successfully!")
