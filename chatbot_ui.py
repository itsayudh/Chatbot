import tkinter as tk
from tkinter import scrolledtext
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Chatbot Logic (Copied from previous response) ---

# Step 1: Prepare Your Data - The FAQ questions and answers
faq_data = {
    # Hiring and Recruitment FAQs
    "What is your hiring process like?": "Our hiring process typically involves a phone screen, a technical interview with the team, and a final interview with a hiring manager. The process can vary by role.",
    "How long does the hiring process take?": "The timeline can vary, but we generally aim to complete the process within 2-4 weeks. We'll keep you updated on your application status.",
    "Do you have any open positions?": "You can view all of our current job openings on our career page on our website.",
    "Can I apply for multiple positions?": "Yes, you can apply for any position you feel you are qualified for. Our recruiting team will review your application for all relevant roles.",
    "Do you offer remote work?": "We offer a mix of in-office, hybrid, and fully remote positions. The working model is specified in the job description.",
    "What programming languages do you use?": "We primarily work with Python, Java, JavaScript (React/Node.js), and C++, but the specific technologies can vary depending on the team and project.",
    "What is your tech stack?": "Our tech stack includes AWS for cloud infrastructure, MongoDB for databases, and Docker for containerization. Specifics depend on the project.",
    "What is the company culture like?": "We pride ourselves on a culture of collaboration, innovation, and respect. We encourage a healthy work-life balance.",
    "What benefits do you offer?": "We offer a comprehensive benefits package that includes health insurance, paid time off, a retirement plan, and more. Details will be provided with a job offer.",

    # Customer/Client FAQs (Building a System)
    "What is your process for building a system?": "Our process starts with a discovery phase to understand your needs, followed by planning, design, development, quality assurance, and deployment. We maintain continuous communication throughout the project.",
    "How much does it cost to build a mobile app?": "The cost of a project depends on many factors, including complexity and features. We provide a detailed, no-obligation quote after our initial discovery call.",
    "How much does it cost to build a website?": "The cost of a project depends on many factors, including complexity and features. We provide a detailed, no-obligation quote after our initial discovery call.",
    "How long does it take to build a system?": "The timeline is highly dependent on the project scope. A simple system might take a few months, while a complex enterprise solution could take over a year.",
    "Can you integrate with our existing systems?": "Yes, we have extensive experience integrating new systems with a wide range of existing software and databases. We will assess your current infrastructure during the discovery phase.",
    "How will we track the project's progress?": "We provide access to a client portal where you can track progress in real-time. We also hold weekly meetings to review milestones and ensure we are aligned with your goals.",
    "What happens after the system is launched?": "We offer post-launch support and maintenance packages to ensure your system runs smoothly. This can include bug fixes, security updates, and new feature development.",
    "Do you provide a warranty?": "Yes, we offer a warranty period to fix any bugs or issues that may arise after the initial launch. The terms of the warranty are specified in our contract.",
}

# Extract questions and answers from the dictionary
questions = list(faq_data.keys())
answers = list(faq_data.values())

# Step 2: Vectorize Your FAQ Questions
vectorizer = TfidfVectorizer().fit(questions)
faq_vectors = vectorizer.transform(questions)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_response(user_question):
    processed_question = preprocess_text(user_question)
    user_vector = vectorizer.transform([processed_question])
    similarities = cosine_similarity(user_vector, faq_vectors)
    best_match_index = similarities.argmax()
    best_match_score = similarities[0, best_match_index]
    
    if best_match_score > 0.4:
        return answers[best_match_index]
    else:
        return "I'm sorry, I couldn't find a direct answer to that. Please try rephrasing your question or contact our support team."

# --- UI Implementation ---

class ChatbotUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IT Company FAQ Chatbot")
        self.root.geometry("600x600")
        self.root.resizable(False, False)

        # Create a container frame for better layout management
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(main_frame, state='disabled', wrap=tk.WORD, bg="#f0f0f0", fg="#333", font=("Helvetica", 12))
        self.chat_display.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.chat_display.tag_config("user", foreground="#007bff", font=("Helvetica", 12, "bold"))
        self.chat_display.tag_config("bot", foreground="#00a080", font=("Helvetica", 12, "bold"))

        # User input area
        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=10)

        self.user_entry = tk.Entry(input_frame, font=("Helvetica", 12), bd=2, relief="groove")
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_entry.bind("<Return>", self.send_message)  # Bind Enter key

        # Send button
        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message, bg="#00a080", fg="white", font=("Helvetica", 10, "bold"))
        self.send_button.pack(side=tk.RIGHT)

        # Initial message from the chatbot
        self.insert_message("bot", "Hello! I'm an IT Company FAQ Chatbot. How can I help you today?")
        
    def insert_message(self, speaker, message):
        """Helper function to insert a message into the chat display."""
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{speaker.title()}: ", speaker)
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)  # Auto-scroll to the bottom

    def send_message(self, event=None):
        """Handles sending a message from the user."""
        user_message = self.user_entry.get().strip()
        if user_message:
            self.insert_message("user", user_message)
            self.user_entry.delete(0, tk.END)
            
            # Check for 'exit' or 'quit' commands
            if user_message.lower() in ["exit", "quit"]:
                self.insert_message("bot", "Goodbye! Feel free to reach out if you have more questions.")
                self.root.after(2000, self.root.destroy) # Close window after 2 seconds
                return

            # Get and display chatbot's response
            bot_response = get_response(user_message)
            self.insert_message("bot", bot_response)

    def run(self):
        """Starts the main UI event loop."""
        self.root.mainloop()

# --- Main Execution Block ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotUI(root)
    app.run()