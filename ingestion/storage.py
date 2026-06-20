import io
import json
import logging
from typing import Optional, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ingestion.models import IngestedDocument
from ingestion.exceptions import AdapterError

logger = logging.getLogger("idip.ingestion.storage")

class S3ParquetWriter:
    """Serializes validated IngestedDocuments as Parquet files and saves them to partitioned S3 paths."""
    
    def __init__(self, s3_client: Optional[Any] = None, bucket_name: str = "idip-data-bucket"):
        self.s3_client = s3_client
        self.bucket_name = bucket_name

    async def write_document(self, doc: IngestedDocument) -> str:
        """
        Converts the IngestedDocument into a Parquet format, partitions by date,
        and uploads to S3. Returns S3 URL string.
        """
        # Formulate partition S3 path matching raw/{year}/{month}/{day}/{source_type}/{doc_id}.parquet
        ts = doc.ingestion_ts
        year = ts.strftime("%Y")
        month = ts.strftime("%m")
        day = ts.strftime("%d")
        
        s3_key = f"raw/{year}/{month}/{day}/{doc.source_type}/{doc.doc_id}.parquet"
        
        try:
            # Map Pydantic model to dict, converting dict/metadata structure to JSON strings for compatibility
            df_dict = {
                "doc_id": [doc.doc_id],
                "ingestion_ts": [doc.ingestion_ts],
                "source_type": [doc.source_type],
                "source_uri": [doc.source_uri],
                "raw_text": [doc.raw_text],
                "raw_bytes": [doc.raw_bytes],
                "byte_size": [doc.byte_size],
                "checksum": [doc.checksum],
                "language": [doc.language],
                "mime_type": [doc.mime_type],
                "page_count": [doc.page_count],
                "metadata": [json.dumps(doc.metadata)]
            }
            
            df = pd.DataFrame(df_dict)
            
            # Serialize to Parquet format in memory
            table = pa.Table.from_pandas(df)
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            parquet_data = buffer.getvalue()
            
            # S3 client write
            if self.s3_client:
                # Expecting standard botocore / boto3 put_object call interface
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=parquet_data,
                    ContentType="application/octet-stream"
                )
                logger.info(f"Parquet file written to s3://{self.bucket_name}/{s3_key}")
            else:
                logger.warning(f"[Mock S3 Write] Bucket: {self.bucket_name}, Key: {s3_key}, Size: {len(parquet_data)} bytes")
                
            return f"s3://{self.bucket_name}/{s3_key}"
            
        except Exception as e:
            raise AdapterError(f"Failed to write parquet file to storage destination: {str(e)}") from e
