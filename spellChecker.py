from spellchecker import SpellChecker


class SpellCheckerWithCorrection:
    def __init__(self):
        self.spell = SpellChecker()

    def correct_sentence(self, input_sentence):
        words = input_sentence.split()
        corrected_sentence = []

        for word in words:
            corrected_word = self.spell.correction(word)
            if corrected_word:
                corrected_sentence.append(corrected_word)
            else:
                corrected_sentence.append(word)

        output_sentence = " ".join(corrected_sentence)
        return output_sentence


def check_spelling(sentence):
    # Example usage:
    spell_checker = SpellCheckerWithCorrection()
    corrected_sentence = spell_checker.correct_sentence(sentence)
    return corrected_sentence
