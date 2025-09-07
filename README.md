# IT Company FAQ Chatbot

A desktop chatbot designed to answer frequently asked questions for both job applicants and potential clients of an IT company. This project is built using Python, leveraging machine learning techniques for intelligent question-answering.

![Python version](https://img.shields.io/badge/Python-3.x-blue.svg)

## 🚀 Features

-   **Dual-Purpose FAQ Knowledge Base:** A comprehensive set of FAQs covering recruitment (hiring process, tech stack, benefits) and client services (project process, pricing, support).
-   **Intelligent Matching:** Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) and Cosine Similarity to accurately match user questions to the most relevant answer.
-   **User-Friendly GUI:** Features a clean and interactive graphical user interface built with the `tkinter` library, providing a seamless chat experience.
-   **Robust Fallback:** Provides a polite, informative response when it cannot find a confident answer to a user's question.

## 🛠️ Prerequisites

To run this chatbot, you need to have Python installed on your system.

-   **Python 3.x:** [Download and install Python](https://www.python.org/downloads/)
-   **Required Libraries:**
    -   `scikit-learn`: For the natural language processing (NLP) and machine learning core.
    -   `tkinter`: For the graphical user interface (GUI). This is usually included with a standard Python installation.

You can install the necessary library using `pip`:

```sh
pip install scikit-learn
