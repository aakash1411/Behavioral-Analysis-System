runfile('/functions.py')
import functions

if __name__ == "__main__":
    sentiment, score, keywords_str = functions.predict("I am really depressed.")
    functions.user_evaluation(sentiment, score, keywords_str)
    
