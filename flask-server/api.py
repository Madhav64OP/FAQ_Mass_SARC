from flask import Flask,request,jsonify
import requests
import pandas as pd
from flask_cors import CORS
import warnings
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app=Flask(__name__)
CORS(app)

json_file={
   "Admissions":[
      {
         "question":"What is the process for admission into Saras AI Institute?",
         "answer":"The admission process at Saras AI Institute typically involves submitting the online application form along with necessary details, followed by a quick pre-Enrollment assessment to evaluate your candidature based on your personal traits and basic communication skills in English."
      },
      {
         "question":"Is there an application fee for applying to Saras AI Institute?",
         "answer":"There is no application fee for applying to any program at Saras"
      },
      {
         "question":"What is pre-enrollment assessment test? How do I prepare for it?",
         "answer":"It is a fully online assessment which takes less than 15 minutes. It is designed to evaluate your personal traits and basic English communication skills. You can take it at the time of filling out the application. It idoes not require any specific preparation"
      },
      {
         "question":"Are there any specific requirements or prerequisites for admission into the programs?",
         "answer":"To be a successful professional in AI, you need to possess basic mathematical proficieny - which can be demonstrated by your math scores in high school or beyond. At Saras, you learn with global peers and faculties and should possess basic communication skills in English. These make up for the basic eligibility critera"
      },
      {
         "question":"When is the deadline for submitting the application?",
         "answer":"The deadline for submitting applications is 5th August 2024"
      }
   ],
   "Curriculum and Faculty":[
      {
         "question":"What is the curriculum like at Saras AI Institute?",
         "answer":"The curriculum at Saras AI Institute helps impart essential technical as well as human skills. We have designed a role-based curriculum that prepares students for one of these in-demand roles: AI/ML Engineer; Data Scientist; Gen AI Engineer. The curriculum is designed to provide a comprehensive understanding of AI principles and practices, including hands-on projects and real-world applications. "
      },
      {
         "question":"What does the program structure look like, and how is the curriculum delivered?",
         "answer":"Each year is divided into 5 semesters which last for 8 weeks each. Our programs feature a mix of recorded and live sessions, allowing for flexibility in learning. "
      },
      {
         "question":"Can you provide more details about the role-based curriculum feature and how it benefits students?",
         "answer":"Our role-based curriculum is designed to provide targeted training and develop specialized skills in students that are highly relevant to their desired job-roles from day one."
      },
      {
         "question":"Do you also conduct LIVE sessions?",
         "answer":"Live sessions are conducted regularly to supplement the recorded content and provide opportunities for interactive learning and Q& A sessions.You get to interact with instructors as well as your dedicated coach for any help that you might need"
      },
      {
         "question":"Can I transfer credits earned at other universities to Saras AI Institute?",
         "answer":"Yes, we evaluate the course that you have taken, if it overlaps with our curriculum and is relevant in today's time, we offer the flexibility of transfering credits"
      },
      {
         "question":"Who are the faculty members at Saras AI Institute?",
         "answer":"The faculty at Saras AI Institute consists of industry professionals who bring the most relevant skills and mentorship for the students to help them prepare for exactly what is needed to succeed in the job roles they are preparing for"
      },
      {
         "question":"Can I connect with mentors outside of class?",
         "answer":"Yes, we encourage mentorship and provide opportunities for students to connect with mentors outside of class through live sessions as well as 24x7 mentor support to help resolve your doubts or queries."
      }
   ],
   "Accreditation & Recognition":[
      {
         "question":"Is Saras AI Institute accredited?",
         "answer":"No, we are not accredited yet. This is our first Enrollment cycle and there is a minimum period before an institute can get accredited. However, we do follow the highest standards in terms of the curriculum and pedagogy for our students to become the top AI professionals"
      },
      {
         "question":"Are the degree programs recognised by the government? ",
         "answer":"Yes, we are a state-approved degree granting institute based in the United States. "
      },
      {
         "question":"Do employers require an accredited degree? ",
         "answer":" An accredited degree is not an absolute requirement from employers. We ensure our students are among the most-skilled and ready individuals to crack the best of jobs"
      }
   ],
   "Career Services":[
      {
         "question":"Does Saras AI Institute offer employment support?",
         "answer":"Yes, we provide comprehensive employment support including job placement services, resume building workshops, and interview preparation."
      },
      {
         "question":"Does Saras have partnerships with employers? ",
         "answer":"Yes, we have partnered with top global companies to recruit our graduating students"
      },
      {
         "question":" Does the university offer internship placement assistance?",
         "answer":"Yes, we assist students in finding internships by connecting them with potential employers and offering guidance on applications and interviews."
      }
   ],
   "Tuition fee and Scholarships":[
      {
         "question":"Does Saras AI Institute offer any scholarships for students? How can I apply for them? ",
         "answer":"Yes, we offer various scholarships to eligible students based on academic merit, financial need, and other criteria. You can apply for scholarships after you're offered admission. Go ahead with filling out the application to check your eligibility."
      },
      {
         "question":" What are the tuition fees for your courses?",
         "answer":"You can find detailed information and breakdown of the fee on 'Programs' page on the website"
      },
      {
         "question":"Are there any payment plans or options available for tuition fees? ",
         "answer":"Yes, we offer flexible payment plans to help students manage their tuition fees. At Saras AI Institute, you can pay your annual tuition fees in 5 installments, before the commencement of every semester."
      },
      {
         "question":"Can I avail financial aid? ",
         "answer":"You currently can't get a federal aid for Saras AI Institute's programs. However, we are partnering with lenders who can help facilitate a loan to help pay the tuition."
      }
   ]
}

def json_to_df(json_data):
    rows = []
    for category, entries in json_data.items():
        for entry in entries:
            row = {'Category': category}
            row.update(entry)
            rows.append(row)
    return pd.DataFrame(rows)

# Convert the JSON structure to DataFrame
raw_df = json_to_df(json_file)

# Display the DataFrame

# prompt: change the |__question with question_text and __answer with answer
# raw_df = pd.DataFrame(faqs)
raw_df = raw_df.rename(columns={'|__question': 'question_text', '|__answer': 'answer'})


q0 = raw_df['question_text'].values[0]
q1 = raw_df['question_text'].values[1]

"""Tokenization
"""
nltk.download('punkt')

word_tokenize(q0)
word_tokenize(q1)

q0_tok = word_tokenize(q0)
q1_tok = word_tokenize(q1)

"""Stop Word Removal -
Removing commonly occuring words
"""

nltk.download('stopwords')
english_stopwords = stopwords.words('english')

", ".join(english_stopwords)

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in english_stopwords]


q0_stp = remove_stopwords(q0_tok)
q1_stp = remove_stopwords(q1_tok)

"""Stemming"""

stemmer = SnowballStemmer('english')

q0_stm = [stemmer.stem(word) for word in q0_stp]

"""Lemmatization"""

small_vect = CountVectorizer()
small_vect.fit(raw_df['question_text'])
small_vect.get_feature_names_out()

"""Trnaform documents into Vectors"""

vectors = small_vect.transform(raw_df['question_text'])

raw_df['question_text'].values[0]

vectors[0].toarray()

vectors.toarray()

"""Configure Count Vectorize Parameters"""
stemmer = SnowballStemmer(language = 'english')

def tokenize(text):
    return [stemmer.stem(word) for word in word_tokenize(text)]

vectorizer = CountVectorizer(lowercase=True,
                            tokenizer=tokenize,
                            stop_words=english_stopwords,
                            max_features=1000)

tokenize('what is really (dealing) here?')

vectorizer.fit(raw_df['question_text'])

vectorizer.get_feature_names_out()

inputs = vectorizer.transform(raw_df['question_text'])

raw_df['question_text'].values[0]


for i in range(inputs.shape[0]):
    print(f"Element {i}: {inputs[i].toarray()}")


def cosine_similarity(vector1, vector2):
    """
    Calculate the cosine similarity between two vectors.
    :param vector1: numpy array of shape (n,) or (1, n)
    :param vector2: numpy array of shape (n,) or (1, n)
    :return: cosine similarity as a float
    """
    # Ensure the vectors are 1D
    vector1 = vector1.flatten()
    vector2 = vector2.flatten()
    # Calculate dot product
    dot_product = np.dot(vector1, vector2)
    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    # Calculate cosine similarity
    cosine_sim = dot_product / (magnitude1 * magnitude2)
    return cosine_sim

# print(inputs[0].shape)
# print(inputs[0].toarray())
# print(type(inputs[0]))
"""similarity is inversely proportional to distance between them
"""
# def find_most_similar_vector(input_vector):
# idx = 0
# distance = 0
# for i in range(inputs.shape[0]):
#     # Reshape input_vector to a 2D array with shape (1, 67)
#     similarity = cosine_similarity(input_vector, inputs[i])
#     if(distance < 1-similarity):
#     distance = 1-similarity
#     idx = i
# return idx
# my_list = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# my_array = np.array(my_list)
# print(my_array)
# print(my_array.shape)
# print(find_most_similar_vector(my_array))
# # !pip install scikit-learn

vec = TfidfVectorizer()
vecs = vec.fit_transform( raw_df['question_text'].apply(lambda x: np.str_(x)))

sim = cosine_similarity(vecs)

def find_most_similar_question_with_answer(question):
    question_vec = vec.transform([question])
    similarity_scores = cosine_similarity(question_vec, vecs)
    most_similar_index = similarity_scores.argmax()
    most_similar_question = raw_df['question_text'].iloc[most_similar_index]
    most_similar_answer = raw_df.answer.iloc[most_similar_index]
    return most_similar_question, most_similar_answer

@app.route('/predict',methods=['POST'])
def predict():
    data=request.get_json()
    input_question=data.get('question')
    if not input_question:
        return jsonify({"error":"No question provided"}),400
    
# # Example usage
# input_question = "HDoes the university help students with finding internships?"
    most_similar_question, most_similar_answer = find_most_similar_question_with_answer(input_question)
# print(f"Most similar question to '{input_question}': {most_similar_question}")
# print(f"Corresponding answer: {most_similar_answer}")
    return jsonify({
        "most_similar_question":most_similar_question,
        "most_similar_answer":most_similar_answer,
    })

if __name__=="__main__":
    app.run(debug=True)
