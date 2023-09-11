import openai
from detection import detect_path
from recognition import check_recognition
from semanticChecker import run_semantic_check
from spellChecker import check_spelling
import tensorflow as tf

tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)


class ApiResponseGenerator:
    def __init__(self, api_key_file_path):
        self.api_key = open(api_key_file_path, "r").read()
        openai.api_key = self.api_key
        self.chat_log = []

    def generate_response(self, user_message):
        if user_message.lower() == "quit":
            print("Exiting program.")
            exit()
        else:
            self.chat_log.append({"role": "user", "content": user_message})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.chat_log
            )
            assistant_response = response['choices'][0]['message']['content']
            chatgpt_response = assistant_response.strip("\n").strip()
            self.chat_log.append({"role": "assistant", "content": chatgpt_response})
            return chatgpt_response


if __name__ == "__main__":
    api_response_generator = ApiResponseGenerator("API_KEY")

    user_message = input("Question: ")
    image_path = input("Image Path(directory name): ")

    chatgpt_response = api_response_generator.generate_response(user_message)

    dir_name = detect_path(image_path)
    sentence = check_recognition(dir_name)
    corrected_sentence = check_spelling(sentence)
    run_semantic_check(corrected_sentence, chatgpt_response)
