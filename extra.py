import random
import string

def random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

def unique_elements(lst):
    return list(set(lst))

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def count_vowels(s):
    return sum(c in 'aeiouAEIOU' for c in s)

def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result

def reverse_words(sentence):
    return ' '.join(word[::-1] for word in sentence.split())

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]