"""Authored, deterministic functional cases; not a public benchmark or broad quality claim."""
import re


def cases():
    groups = {
        "math": [
            ("What is 17 times 19?", "323"),
            ("What is 144 divided by 12?", "12"),
            ("A price of 80 is reduced by 25 percent. What is the new price?", "60"),
            ("Solve 3*x + 7 = 28 for x.", "7"),
            ("What is the sum of the integers from 1 to 20 inclusive?", "210"),
            ("What is the least common multiple of 12 and 18?", "36"),
            ("A train goes 150 km in 2.5 hours. What is its average speed in km/h?", "60"),
            ("What is 2 to the power of 10?", "1024"),
            ("Three boxes each have 24 pens. After giving away 17 pens, how many remain?", "55"),
            ("What is the median of 3, 9, 2, 12, 7?", "7")],
        "qa": [
            ("What is the capital of France?", "Paris"),
            ("Which planet is known as the Red Planet?", "Mars"),
            ("What chemical element has symbol O?", "oxygen"),
            ("How many sides does a hexagon have?", "6"),
            ("Which ocean is the largest on Earth?", "Pacific"),
            ("What is the chemical formula of water?", "H2O"),
            ("Which language uses the keyword SELECT to retrieve database rows?", "SQL"),
            ("In an HTTP response, what numeric status code means Not Found?", "404"),
            ("How many bits are in one byte?", "8"),
            ("What does the S in FIFO stand for? Answer NONE if FIFO has no S.", "NONE")],
        "code": [
            ("In Python, what does print(sum(range(5))) print?", "10"),
            ("In Python, what does print('cache'[::-1]) print?", "ehcac"),
            ("In Python, what does print([x*x for x in range(1,4)]) print?", "[1, 4, 9]"),
            ("In Python, what does print(len(set([1,2,2,3]))) print?", "3"),
            ("In Python, what does print(17 // 5) print?", "3"),
            ("In Python, what does print(17 % 5) print?", "2"),
            ("In Python, what does print(sorted([3,1,2])) print?", "[1, 2, 3]"),
            ("In Python, what does print({'a':1}.get('b',7)) print?", "7"),
            ("In Python, what does print('abcde'[1:4]) print?", "bcd"),
            ("In Python, what does print(all([True, False, True])) print?", "False")],
    }
    result = []
    for category, questions in groups.items():
        for i, (question, answer) in enumerate(questions):
            result.append(dict(id=f"{category}-{i}", category=category,
                prompt=question+" Return only the answer, without explanation or units.", answer=answer))
    for i in range(6):
        # Explicit context facts, not knowledge inferred from the language model.
        filler = "The archive contains unrelated notes about desks, lamps, windows, and chairs. "
        fact = f"The access code of record ZETA-{i} is {7319+i}. "
        result.append(dict(id=f"context-{i}", category="context",
            prompt=(filler*(30+i*10)+fact+filler*(30+i*10)+
                f"What is the access code of record ZETA-{i}? Return only the number."), answer=str(7319+i)))
    return result


def normalize(text):
    text = text.strip().lower().strip("` .\n")
    return re.sub(r"\s+", "", text)


def score(text, case):
    return normalize(text) == normalize(case["answer"])
