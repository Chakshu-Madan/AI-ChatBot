import requests
import os

DOCS_PATH = "documents" 

def fetch_products():
    resp = requests.get("https://fakestoreapi.com/products")
    resp.raise_for_status()
    products = resp.json()

    blocks = []
    for p in products:
        block = (
            f"Product: {p['title']}\n"
            f"Category: {p['category']}\n"
            f"Price: ${p['price']}\n"
            f"Rating: {p['rating']['rate']} out of 5 stars ({p['rating']['count']} reviews)\n"
            f"Description: {p['description']}"
        )
        blocks.append(block)

    content = "\n\n".join(blocks)
    out_path = os.path.join(DOCS_PATH, "products.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Saved {len(products)} products to {out_path}")

if __name__ == "__main__":
    fetch_products()