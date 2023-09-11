import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class SemanticChecker:
    def __init__(self, encoder_path):
        self.encoder = hub.load(encoder_path)
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def embed(self, input):
        return self.encoder(input)

    def calculate_cosine_similarity(self, embeddings):
        similarity_matrix = np.inner(embeddings, embeddings)
        return similarity_matrix

    def extract_and_print_lower_diagonal(self, matrix):
        num_rows, num_cols = matrix.shape
        for i in range(num_rows):
            for j in range(i):
                similarity = round(matrix[i, j].item(), 3)
                print(f'Similarity between two sentences is {similarity}')
                return similarity

    def run_and_print_similarity(self, messages):
        message_embeddings = self.embed(messages)
        similarity_matrix = self.calculate_cosine_similarity(message_embeddings)
        return similarity_matrix


def run_semantic_check(image_ans, chatgpt_response):
    semantic_checker = SemanticChecker('./universal-sentence-encoder_4')
    messages = [image_ans, chatgpt_response]
    similarity_matrix = semantic_checker.run_and_print_similarity(messages)
    print(f'Sentence collected from image: {messages[0]}\nSentence collected from internet: {messages[1]}\n')
    mark = semantic_checker.extract_and_print_lower_diagonal(similarity_matrix.astype(float))
    print(f'Mark for the answer out of 10 is: {mark*10}')


if __name__ == "__main__":
    pass
