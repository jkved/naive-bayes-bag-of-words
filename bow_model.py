import os
import keyword
import re
import json
from collections import Counter
import matplotlib.pyplot as plt
from numpy import std, mean, median


class BagOfWords:
    """
    Class that trains or loads a bag-of-words model for Python and C++ code files,
    applies Laplace smoothing, and then tests on separate files to determine
    how likely they are to be Python or C++ based on keywords.
    """

    py_kwords = set()
    cpp_kwords = set()
    all_kwords = set()

    C_language = 2  # Number of supported languages
    C_kwords = 0
    k_param = 1

    file_count_dict = {}  # Stores the number of files for each type
    trained_dict = {}     # Nested dict storing raw keyword occurrences

    testing_dict = {}     # Holds probabilities after Laplace smoothing
    file_test_dict = {}   # Holds the smoothed probabilities for .py and .cpp files

    def __init__(
            self,
            train_path,
            test_path,
            k_param=1,
            import_prob_bool=True,
            import_name_words='bow_model_occurences.json',
            import_name_lang='bow_model_lang.json'
    ) -> None:
        """
        Initializes the BagOfWords class by either importing existing training data
        (JSON) or training from scratch on the specified training path.

        Args:
            train_path (str): Path to the directory containing training data.
            test_path (str): Path to the directory containing test data.
            k_param (int, optional): The Laplace smoothing parameter. Defaults to 1.
            import_prob_bool (bool, optional): If True, imports pre-trained data.
            import_name_words (str, optional): JSON file name containing word occurrences.
            import_name_lang (str, optional): JSON file name containing language file counts.
        """
        # Load Python and C++ keyword sets
        if isinstance(pyset := PyWords().kwords, set):
            self.py_kwords = pyset
        if isinstance(cppset := CppWords().kwords, set):
            self.cpp_kwords = cppset
        self.all_kwords = self.cpp_kwords | self.py_kwords

        # Either import pre-trained data or train from scratch
        if import_prob_bool:
            print('Imported training data:')
            with open(import_name_words, 'r') as file:
                self.trained_dict = json.load(file)
                print(self.trained_dict)
            with open(import_name_lang, 'r') as file:
                self.file_count_dict = json.load(file)
                print(self.file_count_dict)
        else:
            self._train_for_laplace_smoothing(train_path)
            with open(import_name_words, 'w') as file:
                json.dump(self.trained_dict, file)
            with open(import_name_lang, 'w') as file:
                json.dump(self.file_count_dict, file)

        self.k_param = k_param
        self.train_path = train_path
        self.test_path = test_path

    def test_on_data(self):
        """
        Tests all Python and C++ files in the test directory using the
        trained bag-of-words model and Laplace smoothed probabilities.
        Generates histograms and scatter plots to visualize the results.
        """
        py_files = [f for f in os.listdir(self.test_path) if f.endswith('.py')]
        cpp_files = [f for f in os.listdir(self.test_path) if f.endswith(('.cpp', '.c', '.h'))]

        # Refresh Laplace smoothed probabilities
        self._laplace_smoothing(self.k_param)

        # Calculate probabilities for Python and C++ files
        py_vals, py_tok_counts = self._get_test_probs(
            self.test_path, py_files, self.all_kwords, 'py', 'cpp'
        )
        cpp_vals, cpp_tok_counts = self._get_test_probs(
            self.test_path, cpp_files, self.all_kwords, 'cpp', 'py'
        )

        # Visualization: histograms
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
        ax[0].hist(py_vals)
        ax[0].set(title='py tests')
        ax[1].hist(cpp_vals)
        ax[1].set(title='cpp tests')
        fig.suptitle(f'LS histogram with k={self.k_param}')
        fig.tight_layout()
        fig.savefig(f'graphs/LS_hist_cpp_py_k{self.k_param}.png')

        # Visualization: scatter plots (Laplace smoothed probability vs token count)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
        ax[0].scatter(py_tok_counts, py_vals)
        ax[0].set(title='py tests')
        ax[0].grid()
        ax[1].scatter(cpp_tok_counts, cpp_vals)
        ax[1].set(title='cpp tests')
        ax[1].grid()
        fig.suptitle(f'LS vs token count with k={self.k_param}')
        fig.tight_layout()
        fig.savefig(f'graphs/LS_tokencountt_cpp_py_k{self.k_param}.png')

    def _get_test_probs(self, test_path, files, kwords, tag_main, tag_other):
        """
        Calculates Laplace smoothed probabilities for a list of files, assuming each file
        is of type 'tag_main', in contrast to 'tag_other'.

        Args:
            test_path (str): Path to the test files.
            files (list): List of filenames to process.
            kwords (set): Set of all relevant keywords.
            tag_main (str): Language tag for which the probability is to be calculated ('py' or 'cpp').
            tag_other (str): The other language tag ('cpp' if 'py' is main, and vice versa).

        Returns:
            tuple: A tuple of two lists:
                - vals (list): Probability values for each file being `tag_main`.
                - token_count (list): Number of tokens for each file.
        """
        vals = []
        token_count = []
        fail_iter = self.fail_iterator()

        for f in files:
            filepath = os.path.join(test_path, f)
            tokens = Tokenizer.tokenize_code(filepath, kwords)
            p1 = 1  # Probability of main language
            p2 = 1  # Probability of other language

            for tok in tokens:
                p1 *= self.testing_dict[tag_main][tok]
                p2 *= self.testing_dict[tag_other][tok]

            try:
                # Bayesian probability
                p_main = (
                    (p1 * self.file_test_dict[tag_main]) /
                    (p1 * self.file_test_dict[tag_main] + p2 * self.file_test_dict[tag_other])
                )
                vals.append(p_main)
                token_count.append(len(tokens))
            except ZeroDivisionError:
                next(fail_iter)

        print()
        print(tag_main, ' failed ', it := next(fail_iter) - 1, ' times')
        print('Total test cases executed: ', len(files) - it)
        print('--------------')

        return vals, token_count

    def _train_for_laplace_smoothing(self, train_path):
        """
        Trains the model from scratch using the files in the train_path directory.
        Builds an initial dictionary of keyword occurrences for Python and C++.

        Args:
            train_path (str): Path to the directory containing training files.

        Returns:
            None
        """
        self.trained_dict = {
            tag: {token: 0 for token in self.all_kwords} for tag in ('py', 'cpp')
        }
        self.testing_dict = {
            tag: {token: 0 for token in self.all_kwords} for tag in ('py', 'cpp')
        }

        py_files = [f for f in os.listdir(train_path) if f.endswith('.py')]
        cpp_files = [f for f in os.listdir(train_path) if f.endswith(('.cpp', '.c', '.h'))]

        self._calculate_occurences(self.all_kwords, py_files, train_path, 'py')
        self._calculate_occurences(self.all_kwords, cpp_files, train_path, 'cpp')

    def _calculate_occurences(self, kwords, file_list, folder, tag):
        """
        Counts the occurrences of each keyword in the given list of files, updating
        the trained_dict for the specified language tag.

        Args:
            kwords (set): Set of all relevant keywords (Python + C++).
            file_list (list): List of filenames to process.
            folder (str): Directory containing the files.
            tag (str): Language tag ('py' or 'cpp').

        Returns:
            None
        """
        for f in file_list:
            filepath = os.path.join(folder, f)
            tokens = Counter(Tokenizer.tokenize_code(filepath, kwords))
            for token, count in tokens.items():
                self.trained_dict[tag][token] += count

        self.file_count_dict[tag] = len(file_list)

    def _laplace_smoothing(self, k_param):
        """
        Applies Laplace smoothing to the trained dictionary to convert raw counts
        into probabilities for each keyword. Removes tokens that didn't occur
        in both Python and C++ files.

        Args:
            k_param (int): Laplace smoothing parameter.

        Returns:
            None
        """
        self.C_kwords = len(self.all_kwords)
        for tag in self.trained_dict:
            norm_val = sum(self.trained_dict[tag][token] for token in self.trained_dict[tag])
            self.testing_dict[tag] = {
                token: (self.trained_dict[tag][token] + k_param) /
                       (norm_val + k_param * self.C_kwords)
                for token in self.trained_dict[tag]
                if (self.trained_dict['py'][token] != 0) and (self.trained_dict['cpp'][token] != 0)
            }

        # Update the all_kwords set by removing tokens that didn't occur in both languages
        self.all_kwords = (
            set(self.testing_dict['py'].keys()) | set(self.testing_dict['cpp'].keys())
        )

        # Laplace smooth file probabilities
        total_files = sum(self.file_count_dict[tag] for tag in self.file_count_dict)
        self.file_test_dict = {
            tag: (self.file_count_dict[tag] + k_param) /
                 (total_files + self.C_language * k_param)
            for tag in self.file_count_dict
        }

    @staticmethod
    def fail_iterator():
        """
        A generator that counts how many times a zero division error occurs
        (when no valid tokens are found for a file).

        Yields:
            int: The iteration count of failures.
        """
        count = 0
        while True:
            count += 1
            yield count


class Tokenizer:
    """
    Provides tokenization functionality for .py and .cpp code files.
    Removes string literals, comments, brackets, and splits on non-alphanumeric characters.
    """

    @classmethod
    def tokenize_code(cls, filepath: str, kwords: set):
        """
        Opens a file, reads its contents, removes comments/string literals,
        then tokenizes it by splitting on non-alphanumeric characters,
        returning only the tokens found in the provided set of keywords.

        Args:
            filepath (str): Path to the file to be tokenized.
            kwords (set): Set of keywords to keep (Python, C++ or both).

        Returns:
            list: A list of tokens that are present in the `kwords` set.
        """
        tokens = []
        with open(filepath, 'r') as file:
            code = file.read()

            # Remove entire string literals (both single-quoted and double-quoted)
            code = re.sub(r'(\'[^\']*\'|\"[^\"]*\")', '', code)

            # Remove single-line comments in C++ (//...) and Python (#...)
            code = re.sub(r'\/\/[^\n]*|#[^\n]*', '', code)

            # Remove multi-line comments in C++ (/* ... */)
            # and Python multi-line strings (''' ... ''' or """ ... """)
            code = re.sub(r'\/\*.*?\*\/|(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")',
                          '', code, flags=re.DOTALL)

            # Remove brackets
            code = re.sub(r'(\{|\}|\(|\)|\[|\])', ' ', code)

            # Tokenize by splitting on alphanumeric boundaries
            words = re.findall(r'\b\w+\b', code)

            # Keep only words that match the keyword set
            for word in words:
                word = word.strip()
                if word in kwords:
                    tokens.append(word)

        return tokens


class PyWords:
    """
    Uses Python's standard `keyword.kwlist` as the default set of Python keywords.
    More sophisticated results could be achieved by analyzing actual code constructs.
    """

    kwords = set(keyword.kwlist)

    def __init__(self) -> None:
        print('Python keywords:')
        print(self.kwords)
        print()


class CppWords:
    """
    Uses a predefined set of C++ keywords. This list covers most
    recognized C++ keywords.
    """

    kwords = {
        "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand",
        "bitor", "bool", "break", "case", "catch", "char", "char8_t", "char16_t",
        "char32_t", "class", "compl", "concept", "const", "consteval", "constexpr",
        "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype",
        "default", "delete", "do", "double", "dynamic_cast", "else", "enum",
        "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
        "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept",
        "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private", "protected",
        "public", "register", "reinterpret_cast", "requires", "return", "short",
        "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
        "switch", "synchronized", "template", "this", "thread_local", "throw",
        "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
        "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"
    }

    def __init__(self) -> None:
        print('C++ keywords:')
        print(self.kwords)
        print()


if __name__ == '__main__':
    # Example usage for testing with already-trained data
    for k in [1, 10, 20, 100]:
        print('LS param k = ', k)
        obj_test = BagOfWords(
            train_path='train_data',
            test_path='test_data',
            k_param=k,
            import_prob_bool=True
        )
        obj_test.test_on_data()
