import math
from collections import Counter

def is_prime(num):
    if num < 2: return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def most_common(lst):
    return Counter(lst).most_common(1)[0][0]

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def camel_to_snake(name):
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def swap_case(s):
    return s.swapcase()

def nested_sum(lst):
    return sum(nested_sum(i) if isinstance(i, list) else i for i in lst)

def digit_sum(num):
    return sum(int(d) for d in str(abs(num)))