import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from graphframes import GraphFrame

def main():
    print("Initializing Spark Session with GraphFrames...")
    spark = SparkSession.builder \
        .appName("UPI_Batch_Analytics") \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    # Load data
    print("Loading transaction data...")
    # Using the generated transactions or the IBM dataset slice if you point to it.
    # We will use data/transactions.csv
    edges_df = spark.read.csv("data/transactions.csv", header=True, inferSchema=True)
    
    # Ensure standard GraphFrame column names: 'src', 'dst'
    if "source_account" in edges_df.columns and "target_account" in edges_df.columns:
        edges_df = edges_df.withColumnRenamed("source_account", "src").withColumnRenamed("target_account", "dst")
    
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
