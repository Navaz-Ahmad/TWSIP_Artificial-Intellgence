# Importing necessary libraries
import pandas as pd  # For data handling and manipulation
import spacy  # For Named Entity Recognition (NER) and NLP tasks
import re  # For regex operations such as finding patterns like emails or phone numbers

# Step 1: Data Collection and Preprocessing
# This function loads resumes from a dataset and performs basic cleaning.
# It ensures that the text is properly formatted by removing unnecessary line breaks and extra spaces.
def preprocess_data(data_path):
    # Loading data from a CSV file containing resumes (adjust file path as needed)
    data = pd.read_csv(data_path)
    
    # Removing unwanted newlines and excess whitespace for easier processing
    # 'resume_text' is assumed to be a column in the dataset containing the resume text
    data['cleaned_text'] = data['resume_text'].str.replace(r'\n', ' ').str.replace(r'\s+', ' ', regex=True)
    
    # Returning the cleaned data for further processing
    return data

# Step 2: Named Entity Recognition (NER)
# This function uses a pre-trained spaCy model to identify entities such as names, organizations, and dates.
# NER is crucial in understanding structured data like names and addresses from unstructured resumes.
def named_entity_recognition(text):
    # Loading the English language model from spaCy that is pre-trained for NER tasks
    nlp = spacy.load("en_core_web_sm")
    
    # Processing the input text to identify entities like names, organizations, and dates
    doc = nlp(text)
    
    # Returning a list of entities (such as names, organizations, etc.) with their corresponding labels
    return [(ent.text, ent.label_) for ent in doc.ents]

# Step 3: Parsing Specific Sections of the Resume
# This function uses regular expressions (regex) to extract specific sections like emails and phone numbers from the resume text.
# Regular expressions are great for identifying patterns in the text, such as email formats or phone numbers.
def parse_resume_sections(text):
    # Extracting email addresses from the resume using regex pattern matching
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    
    # Extracting phone numbers (assuming a basic 10-digit format, you can extend this for international numbers)
    phone = re.findall(r'\b\d{10}\b', text)
    
    # Returning both email and phone number(s) found in the resume
    return email, phone

# Step 4: Data Cleaning and Standardization
# This function performs additional cleaning and standardization to ensure uniformity.
# For instance, we might want to standardize date formats or remove OCR (optical character recognition) errors.
def clean_and_standardize(text):
    # Converting common month names to numbers (e.g., "January" to "01") for standardization
    cleaned_text = text.lower().replace('january', '01').replace('february', '02')  # Expand this as needed
    
    # Returning the cleaned and standardized text
    return cleaned_text

# Step 5: Running the Resume Parser
# This function ties everything together: it preprocesses the data, performs NER, parses sections, and cleans the text.
def run_parser(data_path):
    # Step 1: Preprocess the data to remove unwanted formatting issues
    data = preprocess_data(data_path)
    
    # Initializing an empty list to store the final parsed results
    results = []
    
    # Iterating through each resume in the dataset
    for text in data['cleaned_text']:
        # Step 2: Performing NER to extract entities like names, organizations, etc.
        entities = named_entity_recognition(text)
        
        # Step 3: Parsing specific sections like emails and phone numbers
        email, phone = parse_resume_sections(text)
        
        # Step 4: Cleaning and standardizing the text for uniformity
        clean_text = clean_and_standardize(text)
        
        # Storing the results in a structured format (dictionary)
        results.append({
            'entities': entities,  # Extracted entities
            'email': email,        # Extracted email(s)
            'phone': phone,        # Extracted phone number(s)
            'cleaned_text': clean_text  # Standardized resume text
        })
    
    # Returning the final parsed results for all resumes
    return results

# Step 6: Executing the parser
# You can replace 'data_path' with the actual path to your dataset of resumes.
# The parsed_resumes list will contain structured data like entities, emails, and cleaned text for each resume.
data_path = "path_to_your_resumes.csv"  # Provide the actual path to your resume dataset
parsed_resumes = run_parser(data_path)

# Step 7: Displaying the results
# Print the parsed resumes to inspect the output and verify that the parser is working as expected.
print(parsed_resumes)

