import requests


def fetch_wikipedia_article(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "titles": title,
    }
    headers = {"User-Agent": "RAGLearningProject/1.0 (educational project)"}
    response = requests.get(url, params=params, headers=headers)
    print("STATUS CODE:", response.status_code)
    print("RAW RESPONSE (first 300 chars):", response.text[:300])
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    return page.get("extract", "")


titles = ["Dog", "Cat", "Coffee", "Domestication", "Wolf", "Caffeine", "Pet", "Espresso"]

for title in titles:
    print(f"Fetching: {title}")
    text = fetch_wikipedia_article(title)
    filename = f"docs/wiki_{title.lower().replace(' ', '_')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved {len(text)} characters to {filename}")
