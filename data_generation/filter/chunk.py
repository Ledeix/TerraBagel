import pandas as pd

df = pd.read_parquet("top5000_image_pairs_light.parquet")

chunk_size = 500  
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk.to_parquet(
    f"light_chunk_{i//chunk_size}.parquet",
    engine="pyarrow",
    compression="snappy",
    use_dictionary=False
)