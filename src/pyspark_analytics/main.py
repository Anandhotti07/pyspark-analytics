from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

#    Create Spark Session
spark = SparkSession.builder \
    .appName("RandomPySparkExample") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Sample Data
data = [
    (1, "2024-01-01", "Electronics", 2000, "Bangalore"),
    (2, "2024-01-02", "Clothing", 1500, "Mumbai"),
    (3, "2024-01-03", "Electronics", 3000, "Bangalore"),
    (4, "2024-01-04", "Furniture", 4000, "Delhi"),
    (5, "2024-01-05", "Clothing", 1200, "Mumbai"),
    (6, "2024-01-06", "Electronics", 2500, "Delhi"),
]

columns = ["order_id", "order_date", "category", "amount", "city"]

df = spark.createDataFrame(data, columns)

#  Convert to proper date type
df = df.withColumn("order_date", to_date(col("order_date"), "yyyy-MM-dd"))

#           Basic Aggregation
agg_df = df.groupBy("category") \
           .agg(sum("amount").alias("total_sales"),
                avg("amount").alias("avg_sales"),
                count("*").alias("order_count"))

# 2️⃣ Wind    ow Function - Running Total by Category
window_spec = Window.partitionBy("category") \
                    .orderBy("order_date") \
                    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df = df.withColumn("running_total",
                   sum("amount").over(window_spec))

# 3️⃣ Ranking within Category
rank_window = Window.partitionBy("category").orderBy(desc("amount"))

df = df.withColumn("rank_in_category",
                   dense_rank().over(rank_window))

# 4️⃣ Filtering High Value Orders
high_value_df = df.filter(col("amount") > 2000)

# 5️⃣ Join Example (self join for same city comparison)
df_alias1 = df.alias("a")
df_alias2 = df.alias("b")

join_df = df_alias1.join(
    df_alias2,
    (col("a.city") == col("b.city")) &
    (col("a.order_id") != col("b.order_id")),
    "inner"
).select(
    col("a.order_id").alias("order1"),
    col("b.order_id").alias("order2"),
    col("a.city")
)

# Show Results
print("Aggregated Data:")
agg_df.show()

print("With Running Total and Rank:")
df.show()

print("High Value Orders:")
high_value_df.show()

print("Self Join Result:")
join_df.show()
