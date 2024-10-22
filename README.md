# **SARC Round-2 FAQ Portal : Team Mars**

## **Project Overview**
This project implements a question-answering system using Natural Language Processing (NLP) techniques, specifically the TF-IDF (Term Frequency-Inverse Document Frequency) method, combined with cosine similarity. The system answers user queries by finding the most similar pre-defined question and providing the corresponding answer. The project has the following components:
- A backend built with Flask that processes and handles API requests.
- A front-end website hosted on GitHub Pages to interact with the user.
- The API is deployed on Render for production usage.

## **How to use**
- **Live Version**: You can access the live version of the project through this link:  
  [FAQ Portal - Team Mars](https://madhav64op.github.io/FAQ_SARC_Team_Mass/)

- **To use this code locally**:
  1. **Clone the Repository**:  
     Run the following command in your terminal:  
     ```bash
     git clone https://github.com/madhav64op/FAQ_SARC_Team_Mass.git
     ```

  2. **Navigate to the Project Directory**:  
     ```bash
     cd FAQ_SARC_Team_Mass
     ```

  3. **Open the Project in Visual Studio Code**:  
     Open Visual Studio Code and select **File > Open Folder**, then choose the `FAQ_SARC_Team_Mass` folder.

  4. **Run the Live Server**:  
     - Install the **Live Server** extension from the VS Code marketplace if you haven't already.
     - Once installed, right-click on the `index.html` file in the Explorer panel and select **Open with Live Server**.

  5. **Access the Application**:  
     The application will open in your default web browser at `http://127.0.0.1:5500/` (or another port as specified by Live Server).

Now, you can interact with the FAQ Portal locally!


## **What we did**
1. **Text Preprocessing and Vectorization**: 
   - The project utilizes `TfidfVectorizer` from `sklearn` to convert input questions into vector representations.

2. **Cosine Similarity**: 
   - To determine the similarity between user-provided questions and the dataset, `cosine_similarity` is applied to find the most relevant answer.

3. **Flask-based API**: 
   - The Flask framework processes API requests and returns the most similar question-answer pair.

4. **Cross-Origin Resource Sharing (CORS)**: 
   - Flask-CORS enables the front-end on GitHub Pages to interact with the backend API hosted on Render. We encountered some issues with CORS licenses while integrating it with our React front-end.

5. **React Integration**: 
   - We developed the front-end using React to create a user-friendly interface for submitting questions and receiving answers.

6. **RNN-Bi-directional LSTM Model**: 
   - Initially, we attempted to implement an RNN-Bi-directional LSTM model for more advanced question-answering capabilities. However, we faced several challenges, including model training issues and performance constraints, which led us to adopt the current approach using TF-IDF and cosine similarity.


## **System Workflow**
1. **Input**: The user submits a question through the front-end interface.
2. **Backend Processing**:
   - The submitted question is vectorized using `TfidfVectorizer`.
   - The cosine similarity between the input question and predefined questions is computed.
   - The most similar question and its corresponding answer are retrieved.
3. **Output**: The backend sends the most similar question and its answer as a JSON response to the front-end, which displays it to the user.

---

## **Code Structure**

### **Backend** (Flask-based API)
- **File: `api.py`**
  - **Libraries Used**:
    - `pandas`: For handling the dataset in a tabular format.
    - `TfidfVectorizer`: To vectorize text questions.
    - `cosine_similarity`: To calculate the similarity between user input and the dataset questions.
    - `Flask`, `CORS`: To handle API requests and manage cross-origin resource sharing.
  - **Key Functions**:
    - `json_to_df`: Converts a JSON object to a Pandas DataFrame.
    - `find_most_similar_question_with_answer`: Computes cosine similarity and returns the most similar question-answer pair.
    - `/predict`: Flask route to process POST requests containing a question and return the predicted answer.
  - **Deployment**: Hosted on Render platform.
  
### **Front-End**
- **Website**: The front-end interface is a simple website that allows users to input their questions. It sends the question to the Flask API and displays the answer on the page.
- **Deployment**: The website is hosted on **GitHub Pages**.


