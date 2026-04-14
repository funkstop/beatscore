from beatscore import run_digest

sources = {
    "reuters": "http://feeds.reuters.com/reuters/topNews",
    "bbc": "http://feeds.bbci.co.uk/news/rss.xml",
    "nyt": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "rollingStone": "https://www.rollingstone.com/feed/"
    }


run_digest(sources, "news", output_dir="./output")
