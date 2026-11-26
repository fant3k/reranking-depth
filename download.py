from pathlib import Path
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

DATA = Path("data")
AMAZON = "https://media.githubusercontent.com/media/amazon-science/esci-data/7916cdf6ab75a462e77f20ab40428a10923998d5/shopping_queries_dataset/shopping_queries_dataset_"
WAYFAIR = "https://raw.githubusercontent.com/wayfair/WANDS/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/dataset/"
MODEL = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/233902d25c440f23af6f7d6e94d2946bac0bee0a/"
FILES = [
    ("examples.parquet", AMAZON+"examples.parquet", "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a"),
    ("products.parquet", AMAZON+"products.parquet", "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265"),
    ("product.csv", WAYFAIR+"product.csv", "a415f7295dc0d90b3163ef606da611a4e9ddeca8"),
    ("query.csv", WAYFAIR+"query.csv", "acb3c1f4cf1b775f494225afec79ec2163fc3e03"),
    ("label.csv", WAYFAIR+"label.csv", "2c09a0d7b1f77f6040deb5873b5a90b8ef1e3ae9"),
    ("model.onnx", MODEL+"onnx/model.onnx", "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a"),
    ("tokenizer.json", MODEL+"tokenizer.json", "688882a79f44442ddc1f60d70334a7ff5df0fb47"),
]

def digest(path, expected):
    sha = hashlib.sha256() if len(expected)==64 else hashlib.sha1()
    if len(expected)==40:
        sha.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        while chunk := f.read(4<<20):
            sha.update(chunk)
    return sha.hexdigest()

def fetch(item):
    name,url,expected=item
    dest=DATA/name
    if dest.exists() and digest(dest,expected)==expected:
        return
    tmp=dest.with_suffix(".partial")
    subprocess.run(["curl","-fsSL","--retry","4","--retry-all-errors","--connect-timeout","20","--max-time","1500","-o",str(tmp),url],check=True)
    if digest(tmp,expected)!=expected:
        raise ValueError(f"Invalid checksum for {name}")
    tmp.replace(dest)
    print(name, dest.stat().st_size, flush=True)

if __name__=="__main__":
    DATA.mkdir(exist_ok=True)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(fetch,FILES))
