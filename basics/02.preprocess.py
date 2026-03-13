import re 

def extraction(text:str):
    text = re.sub(r'\n','',text)
    text = re.sub(r'[\U0001F300-\U0001F9FF]','',text)
    sentances = re.split(r'(?<=[.!?])\s+',text.strip())
    print(sentances)
    
    percentages = []
    dates = []
    entities = []
    amounts = []
    countries=[]
    emails = []
    urls=[]
    for sentance in sentances:
        percentage = re.findall(r'\d+(?:\.\d{2})?%',sentance)
        if percentage:
            percentages.extend(percentage)
        date = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',sentance)
        if date:
            dates.extend(date)
        date = re.findall(r'(?:Jan|Feb|Mar)\s+\d{1,2},\s+\d{2,4}\s+',sentance)
        if date:
            dates.extend(date)
        entitie = re.findall(r'[A-Z][A-Za-z]+\s+(?:Inc.|Corp|Agreement)',sentance)
        if entitie:
            entities.extend(entitie)
        amount = re.findall(r'(?:\$|USD)\s*\d+,\d+,\d+\s*(?:USD)?',sentance)
        if amount:
            amounts.extend(amount)
        country = re.findall(r'(?:United States|Canada|United Kingdom|Germany|France|Japan)\b',sentance)
        if country:
            countries.extend(country)
        email = re.findall(r'[\w\.]+@\w+\.\w+',sentance)
        if email:
            emails.extend(email)
        url = re.findall(r'https?://\w+\.\w+',sentance)
        if url:
            urls.extend(url)
    print(percentages)
    print(dates)
    print(entities)
    print(amounts)
    print(countries)
    print(emails)
    print(urls)

contract = """
    This Distribution Agreement ("Agreement") is entered into between Apple Inc.
    and Partner Corp effective Jan 1, 2025 1/1/2025. The territory covered under this
    agreement includes the United States, Canada, United Kingdom, Germany, France,
    and Japan. The content licensed includes all music catalog items, podcasts,
    and audiobook titles raj@gmail.com. The royalty rate shall be 70% of net revenue for music
    and 50% for podcast content. Payment terms are net-30 from the end of each
    calendar quarter. The total licensing fee is $2,500,000 USD per year.
    This agreement shall remain in effect for a period of 3 years from the
    effective date.https://apple.com Apple reserves the right to adjust pricing with 90 days
    written notice.
    """
extraction(contract)