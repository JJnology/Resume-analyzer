from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_text_from_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def get_keywords(text, top_n=10):

    vectorizer = TfidfVectorizer(stop_words='english')

    tfidf = vectorizer.fit_transform([text])

    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return [word for word, score in sorted_scores[:top_n]]

def get_similarity(text1, text2):

    vectorizer = TfidfVectorizer(stop_words='english')

    tfidf = vectorizer.fit_transform([text1, text2])

    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])

    return similarity[0][0]

def display_results(resume_text, job_description_text):

    resume_keywords = get_keywords(resume_text)

    job_description_keywords = get_keywords(job_description_text)

    similarity_score = get_similarity(resume_text, job_description_text)

    print("Resume vs Job Description Similarity Score:", similarity_score)
    print("\nTop skills in Resume:", resume_keywords)
    print("\nTop skills in Job Description:", job_description_keywords)

    missing_keywords = [word for word in job_description_keywords if word not in resume_keywords]

    print("\nMissing skills from Resume:", missing_keywords)

resume_text = load_text_from_file('resume.txt')
job_description_text = load_text_from_file('job_description.txt')

display_results(resume_text, job_description_text)