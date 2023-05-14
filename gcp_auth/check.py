### 인증 확인
from google.cloud import storage

storage_client = storage.Client()
buckets = list(storage_client.list_buckets())

print(buckets) # 결과 => [<Bucket: 버킷 이름>]