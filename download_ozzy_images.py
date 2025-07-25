import os
from icrawler.builtin import GoogleImageCrawler

# Updated base directory to match your folder
base_dir = r"F:\Fun project\Ozzy Fashion Classifier"

# Define keyword → folder mapping
keywords = {
    "bat_cape_era": "Ozzy Osbourne bat cape outfit",
    "glam_metal_80s": "Ozzy Osbourne 1980s glam metal outfit",
    "mtv_ozzy": "Ozzy Osbourne MTV show screenshots",
    "modern_ozzy": "Ozzy Osbourne 2020 sunglasses",
    "award_show_outfits": "Ozzy Osbourne red carpet look"
}

# Function to download images
def download_images(label, keyword, max_images=50):
    save_dir = os.path.join(base_dir, label)
    os.makedirs(save_dir, exist_ok=True)
    crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
    crawler.crawl(keyword=keyword, max_num=max_images)
    print(f"✅ Downloaded {max_images} images for '{label}' → {save_dir}")

# Download for each label
for label, keyword in keywords.items():
    download_images(label, keyword, max_images=50)
