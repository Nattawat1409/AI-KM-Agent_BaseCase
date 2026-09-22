# check_sparse.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE"])

# 1) เป็น integrated embedding ไหม (ถ้ามี field `embed` แปลว่ามี model ผูกกับ index)
print(pc.describe_index("cimie-km-th-sparse"))

idx = pc.Index("cimie-km-th-sparse")
stats = idx.describe_index_stats()
print(f"namespace and number of vectors: {stats}")                                   # namespace และจำนวน vector

# 2) ดึง vector ตัวอย่าง (ปริ้นท์เฉพาะ "ชื่อ key" ของ metadata ไม่ปริ้นท์เนื้อหาเอกสาร)
ns = next(iter(stats.namespaces), "")          # namespace แรก ถ้าไม่มีใช้ ""
first_ids = next(iter(idx.list(namespace=ns)))
res = idx.fetch(ids=first_ids[:1], namespace=ns)
for vid, v in res.vectors.items():
    sv = v.sparse_values
    print("id:", vid)
    print("metadata keys:", list((v.metadata or {}).keys()))
    print("nnz:", len(sv.indices), "| indices[:10]:", sv.indices[:10], "| values[:10]:", sv.values[:10])