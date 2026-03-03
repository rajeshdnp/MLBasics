"""
Q17 — Feature Extraction Pipeline for Multilingual Contracts [CRITICAL]
Target time: 25 min | CoderPad-safe (Python stdlib only)

APPROACH (say this first 60 seconds):
"I'll build extract_features(text, language) that runs a three-stage pipeline:
(1) sentence segmentation, (2) keyword extraction via frequency-based scoring after
stopword removal, (3) entity recognition using regex patterns for dates, currencies,
territories, and party names. All CoderPad-safe — pure regex and string operations.
The pipeline is designed to be extensible for new languages."

CORE MATH:
- Keywords: frequency-based ranking after stopword removal
- Entities: regex pattern matching for dates, currencies, territories

TIME: O(n * patterns) | SPACE: O(entities)
"""

import re
from collections import Counter

STOPWORDS = {
    'en': {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to',
           'for', 'of', 'and', 'or', 'but', 'not', 'this', 'that', 'with', 'by',
           'from', 'as', 'be', 'have', 'has', 'had', 'will', 'shall', 'may', 'can',
           'its', 'it', 'all', 'any', 'each', 'such', 'than', 'other'},
    'ja': set(),
}

CURRENCY_PATTERNS = [
    r'[\$\u20ac\u00a3\u00a5]\s*[\d,]+(?:\.\d{2})?',
    r'[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY)',
    r'(?:USD|EUR|GBP|JPY)\s*[\d,]+(?:\.\d{2})?',
]

DATE_PATTERNS = [
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    r'\b(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
    r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
]

TERRITORIES = [
    'United States', 'Canada', 'United Kingdom', 'Germany', 'France',
    'Japan', 'China', 'Australia', 'Brazil', 'India', 'South Korea',
    'Mexico', 'Italy', 'Spain', 'US', 'UK', 'EU', 'APAC', 'EMEA', 'LATAM',
]

PARTY_PATTERNS = [
    r'(?:between|by)\s+([A-Z][A-Za-z\s\.]+?(?:Inc|Corp|LLC|Ltd|Co|GmbH|SA|AG)\.?)',
]
PARTY_BLACKLIST = {'Agreement', 'Appendix', 'Section', 'Article'}

PERCENTAGE_PATTERN = r'\b\d+(?:\.\d+)?%'
DURATION_PATTERNS = [
    r'\b\d+\s+(?:year|month|day|week)s?\b',
    r'\b(?:net-?\d+)\b',
]


def segment_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_keywords(text: str, language: str = 'en', top_k: int = 10) -> list:
    stops = STOPWORDS.get(language, STOPWORDS['en'])
    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
    filtered = [t for t in tokens if t not in stops and len(t) > 2]
    return [word for word, _ in Counter(filtered).most_common(top_k)]


def extract_entities(text: str) -> dict:
    text_clean = re.sub(r'\s+', ' ', text)
    entities = {'dates': [], 'currencies': [], 'territories': [],
                'parties': [], 'percentages': [], 'durations': []}

    for pattern in DATE_PATTERNS:
        entities['dates'].extend(re.findall(pattern, text_clean, re.IGNORECASE))

    for pattern in CURRENCY_PATTERNS:
        entities['currencies'].extend(re.findall(pattern, text_clean))

    text_lower = text_clean.lower()
    for territory in TERRITORIES:
        if territory.lower() in text_lower:
            entities['territories'].append(territory)

    for pattern in PARTY_PATTERNS:
        matches = re.findall(pattern, text_clean)
        entities['parties'].extend([m.strip() for m in matches
                                    if m.strip() not in PARTY_BLACKLIST])

    entities['percentages'] = re.findall(PERCENTAGE_PATTERN, text_clean)

    for pattern in DURATION_PATTERNS:
        entities['durations'].extend(re.findall(pattern, text_clean, re.IGNORECASE))

    # deduplicate preserving order
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
    return entities


def extract_features(text: str, language: str = 'en') -> dict:
    """Full feature extraction pipeline for contract text."""
    if not text or not text.strip():
        return {'sentences': [], 'keywords': [], 'entities': {},
                'language': language, 'num_sentences': 0}

    sentences = segment_sentences(text)
    keywords = extract_keywords(text, language)
    entities = extract_entities(text)

    return {
        'language': language,
        'num_sentences': len(sentences),
        'sentences': sentences,
        'keywords': keywords,
        'entities': entities,
    }


# === TEST ===
if __name__ == "__main__":
    contract = """
    This Distribution Agreement ("Agreement") is entered into between Apple Inc.
    and Partner Corp effective January 1, 2025. The territory covered under this
    agreement includes the United States, Canada, United Kingdom, Germany, France,
    and Japan. The content licensed includes all music catalog items, podcasts,
    and audiobook titles. The royalty rate shall be 70% of net revenue for music
    and 50% for podcast content. Payment terms are net-30 from the end of each
    calendar quarter. The total licensing fee is $2,500,000 USD per year.
    This agreement shall remain in effect for a period of 3 years from the
    effective date. Apple reserves the right to adjust pricing with 90 days
    written notice.
    """

    result = extract_features(contract, language='en')
    print(f"Language: {result['language']}")
    print(f"Sentences: {result['num_sentences']}")
    print(f"Keywords: {result['keywords']}")
    print(f"Dates: {result['entities']['dates']}")
    print(f"Currencies: {result['entities']['currencies']}")
    print(f"Territories: {result['entities']['territories']}")
    print(f"Percentages: {result['entities']['percentages']}")
    print(f"Durations: {result['entities']['durations']}")
    print(f"Parties: {result['entities']['parties']}")

    # Edge case
    empty_result = extract_features("")
    assert empty_result['num_sentences'] == 0
    print("\nAll tests passed!")
