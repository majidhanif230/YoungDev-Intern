import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
import random
import datetime
import time

# Define a dictionary for storing appointments
appointments = {}

# Define a dictionary for storing frequently asked questions and answers
faq = {
    "What is the currency of Pakistan?": "The currency of Pakistan is the Pakistani Rupee (PKR).",
    "How do I get a SIM card in Pakistan?": "You can get a SIM card in Pakistan by visiting a local cellular network service provider's outlet, providing your identification documents, and completing the necessary paperwork.",
    "What is the national language of Pakistan?": "The national language of Pakistan is Urdu, while English is also widely used for official and business purposes.",
    "How do I travel within Pakistan?": "You can travel within Pakistan by various means including buses, trains, and domestic flights. Each major city usually has its own bus terminals, railway stations, and airports.",
    "What is the traditional cuisine of Pakistan?": "The traditional cuisine of Pakistan includes dishes like biryani, karahi, kebabs, and various types of bread such as naan and roti.",
    "What are the major tourist attractions in Pakistan?": "Major tourist attractions in Pakistan include historical sites such as the Badshahi Mosque and Lahore Fort in Lahore, the Faisal Mosque in Islamabad, the ancient city of Mohenjo-Daro, and the scenic beauty of northern areas like Hunza Valley and Swat Valley.",
    "What is the time zone in Pakistan?": "Pakistan Standard Time (PST) is the time zone used in Pakistan, which is UTC+5 hours.",
    "How do I apply for a Pakistani visa?": "You can apply for a Pakistani visa through the nearest Pakistani embassy or consulate in your country. The specific requirements and application process may vary depending on your nationality.",
    "What is the literacy rate in Pakistan?": "The literacy rate in Pakistan is estimated to be around 60-65%, with significant variation between urban and rural areas and between males and females.",
    "How do I access healthcare in Pakistan?": "You can access healthcare in Pakistan through public hospitals, private hospitals, clinics, and pharmacies. It's advisable to have health insurance or sufficient funds to cover medical expenses.",
    "What are the popular sports in Pakistan?": "Cricket is the most popular sport in Pakistan, followed by field hockey and football (soccer). Other traditional sports include kabaddi and squash.",
    "What is the dress code in Pakistan?": "The dress code in Pakistan varies depending on factors such as region, culture, and occasion. In general, modest attire is preferred, especially in more conservative areas.",
    "What are the major festivals celebrated in Pakistan?": "Major festivals celebrated in Pakistan include Eid al-Fitr and Eid al-Adha, which are religious festivals celebrated by Muslims, as well as cultural festivals such as Basant (kite festival) and Shandur Polo Festival.",
    "What is the education system like in Pakistan?": "The education system in Pakistan is divided into primary, secondary, and tertiary levels. Primary education is compulsory and free in government schools, while private schools and universities also play a significant role.",
    "What is the weather like in Pakistan?": "The weather in Pakistan varies from region to region. Generally, Pakistan experiences hot summers and cold winters, with monsoon rains in the summer months and occasional snowfall in the northern areas during winter."
}

# Define a list of responses to user queries
responses = [
    'I\'m not sure I understand. Can you please rephrase?',
    'That\'s a great question! I\'m not sure I have the answer, though.',
    'I\'m still learning, but I\'ll do my best to help!',
    'I\'m not sure what you mean. Can you please provide more context?',
    'That\'s an interesting question! I\'ll have to think about it.',
    # Add more responses here...
]

# Define a list of greetings
greetings = ['hello!', 'hi there!', 'greetings!', 'hey!', 'nice to see you!', 'howdy!']

# Define a list of farewells
farewells = ['goodbye!', 'see you later!', 'farewell!', 'until next time!', 'take care!', 'bye!']


# Define a function to generate a random greeting
def get_greeting():
    return random.choice(greetings)


# Define a function to generate a random farewell
def get_farewell():
    return random.choice(farewells)


# Define a function to book an appointment
def book_appointment():
    while True:
        date = input("What date would you like to book the appointment? (Please enter in YYYY-MM-DD format): ")
        try:
            year, month, day = map(int, date.split('-'))
            if year != 2024 or not (1 <= month <= 12) or not (1 <= day <= 30):
                print(
                    "Please enter a valid date within the year 2024 with a month between 1 and 12 and a day between 1 and 30.")
            else:
                break
        except ValueError:
            print("Invalid date format. Please enter the date in YYYY-MM-DD format.")

    while True:
        time = input("What time would you like to book the appointment? (Please enter in HH:MM format): ")
        try:
            hour, minute = map(int, time.split(':'))
            if not (0 <= hour <= 23) or not (0 <= minute <= 59):
                print("Please enter a valid time with hour between 0 and 23 and minute between 0 and 59.")
            else:
                break
        except ValueError:
            print("Invalid time format. Please enter the time in HH:MM format.")

    thing = input("What would you like to book? (e.g., meeting, appointment, hotel, car): ")
    return f'Appointment booked for {date} at {time} for {thing}.'


# Define a function to cancel an appointment
def cancel_appointment(date, time):
    if (date, time) in appointments:
        del appointments[(date, time)]
        return f'Appointment canceled for {date} at {time}.'
    return f'No appointment found for {date} at {time}.'


# Define a function to answer user queries
# Define a function to answer user queries
def answer_query(query):
    # Tokenize and lowercase the query
    tokens = nltk.word_tokenize(query.lower())

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]

    # Check if the query is a greeting
    if any(word in lemmas for word in ['hello', 'hy', 'hi', 'hey', 'greetings', 'howdy']):
        return get_greeting()

    # Check if the query is a farewell
    if any(word in lemmas for word in ['goodbye', 'bye', 'farewell']):
        return get_farewell()

    # Check if the query is a FAQ
    for question, answer in faq.items():
        if all(lemma in question.lower() for lemma in lemmas):
            return answer

    # Check if the query contains words related to booking an appointment
    if any(word in lemmas for word in ['book', 'appoint', 'schedule']):
        return book_appointment()

    # Check if the query is about canceling an appointment
    if any(word in lemmas for word in ['cancel', 'delete']):
        # Extract the date and time from the query
        date = None
        time = None
        for token in tokens:
            if token.isdigit():
                date = token
            elif ':' in token:
                time = token
        if date and time:
            return cancel_appointment(date, time)

    # Perform named entity recognition (NER) to extract entities
    entities = ne_chunk(pos_tag(tokens))

    # Check if the query contains a specific entity
    for entity in entities:
        if hasattr(entity, 'label'):
            if entity.label() == 'PERSON':
                return f'Hello, {entity[0][0]}!'
            elif entity.label() == 'DATE':
                return f'The date is {datetime.date.today()}.'
            # Add more entity recognition here...

    # Return a random response if the query is not recognized
    return random.choice(responses)



# Define a function to get a sample question from FAQ or appointments
def get_sample_question():
    choice = random.choice(["FAQ"])
    if choice == "FAQ":
        return random.choice(list(faq.keys()))


# Define the main function to interact with the user
def main():
    print("Welcome to the chatbot!")
    print("Sample Questions:")
    for _ in range(5):
        print("-", get_sample_question())
    print("-", "book a car ")

    while True:
        # Prompt the user for input
        user_input = input("\nUser: ").lower()  # Convert input to lowercase
        if user_input == "quit":
            print("Chatbot: Goodbye!")
            break
        response = answer_query(user_input)
        print("Chatbot:", response)


if __name__ == "__main__":
    main()
