import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from graphframes import GraphFrame

def main():
    print("Initializing Spark Session with GraphFrames...")
    spark = SparkSession.builder \
        .appName("UPI_Batch_Analytics") \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "2g") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    # Load data from the real IBM AML dataset
    print("Loading IBM transaction data...")
    ibm_path = "/kagglehub/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/versions/8/HI-Small_Trans.csv"
    
    # The IBM CSV has duplicate 'Account' columns (sender and receiver) which breaks Spark's default inferSchema.
    # We will enforce a custom schema.
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
    schema = StructType([
        StructField("Timestamp", StringType(), True),
        StructField("FromBank", StringType(), True),
        StructField("src", StringType(), True),
        StructField("ToBank", StringType(), True),
        StructField("dst", StringType(), True),
        StructField("AmountReceived", DoubleType(), True),
        StructField("ReceivingCurrency", StringType(), True),
        StructField("AmountPaid", DoubleType(), True),
        StructField("PaymentCurrency", StringType(), True),
        StructField("PaymentFormat", StringType(), True),
        StructField("IsLaundering", IntegerType(), True)
    ])
    
    edges_df = spark.read.csv(ibm_path, header=True, schema=schema)
    
    # The local Docker container is hard-limited to ~2.4GB of JVM heap memory by WSL2 defaults.
    # PageRank on 5.07 million edges (which requires heavy shuffle joins) triggers OutOfMemoryError.
    # To demonstrate the architecture locally, we sample 10% of the graph (approx 500,000 edges).
    print("Sampling 10% of the 5-million edge graph to fit in local memory limits...")
    edges_df = edges_df.sample(0.1, seed=42)
    
    # Create vertices DataFrame from unique src and dst
    src_df = edges_df.select(col("src").alias("id"))
    dst_df = edges_df.select(col("dst").alias("id"))
    vertices_df = src_df.union(dst_df).distinct()
    
    print(f"Graph loaded. Vertices: {vertices_df.count()}, Edges: {edges_df.count()}")
    
    g = GraphFrame(vertices_df, edges_df)
    
    print("--- Running Degree Centrality ---")
    start_time = time.time()
    degrees = g.degrees
    degrees.orderBy(col("degree").desc()).show(5)
    print(f"Degree Centrality computed in {time.time() - start_time:.2f} seconds.")
    
    print("--- Running PageRank ---")
    start_time = time.time()
    # Run PageRank for a fixed number of iterations for predictable performance
    pr = g.pageRank(resetProbability=0.15, maxIter=5)
    
    print("Top accounts by PageRank:")
    pr.vertices.orderBy(col("pagerank").desc()).show(5)
    print(f"PageRank computed in {time.time() - start_time:.2f} seconds.")
    
    spark.stop()

if __name__ == "__main__":
    main()
