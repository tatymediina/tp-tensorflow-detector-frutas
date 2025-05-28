from icrawler.builtin import GoogleImageCrawler

# Lista de frutas que queremos descargar
frutas = ["manzana", "banana", "pera", "naranja", "uva"]

for fruta in frutas:
    google_crawler = GoogleImageCrawler(storage={'root_dir': f'frutas/{fruta}'})
    google_crawler.crawl(keyword=fruta, max_num=2, file_idx_offset=0)
