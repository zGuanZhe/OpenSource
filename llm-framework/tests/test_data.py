import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.tokenizer import SimpleTokenizer
from data.preprocess import clean_text

class TestData(unittest.TestCase):
    def test_tokenizer(self):
        tokenizer = SimpleTokenizer(vocab_size=256)
        text = "hello"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(text, decoded)

    def test_preprocess(self):
        raw = "  This \n is a   test.  "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "This is a test.")

if __name__ == "__main__":
    unittest.main()
