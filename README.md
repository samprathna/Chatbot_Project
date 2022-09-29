# Data

The dataset is a dataset downloaded from kaggle.com. It is a dataset containing customer service queries "utterances" and their corresponding query class "intent". From that data, I will be training my chatbot to classify the embedded utterances (vectors) and respond with the proper reply (a print). For each utterance class is attributed one response.

# Project in steps:
1- Creating a container with replies for the 10 selected classes.
2- For the selected classes' utterances, we will pre-process the data so it becomes encodable.
3- I encoded (padded_sequence of padded, sequenced, tokenized) the data. We can also use other types of embedding methods. Namely TFIDF Vectorizer to vectorize each of our utterances. The method used here is arbitrary. The main focus of this project was to deploy a functioning chatbot, and not maximizing the model's precision.
4- I split the data into training/testing datasets and then we train the model
5- I look at the performance of our model as well as its accuracy and model loss curve. Adjust our model/data embedding methods if necessary.
6- Test the model, save the model and its depedencies (if any) as pickle files.
7- Load the model and deploy it using the Flask framework on a local machine. (See "Final_Project_Main_NoteBook.ipynb")

# Chatbot use in steps:
1- User input:                           "How do I talk to someone" <br>
2- Pre-processing (NLTK)                 -> input becomes "how talk someone" <br>
3- Encoding steps                        -> \["how", "talk", "someone"] <br>
4- Encoding steps                        -> \[0,0,0,0,0,42,61,23] <br>
5- Input the vector/embedded_words       -> Goes into the neural network's blackbox <br>
6- Classification                        -> 80\% confidence that the class for "How do I talk to someone" is 3 <br>
7- Bot responds with reply for class = 3 -> "To contact a human agent, ..."

![alt text](https://github.com/samprathna/Chatbot_Project/blob/main/images/example-using-chatbot.png)
