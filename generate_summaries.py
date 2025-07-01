from summarization import cluster_summarizer

print("🤖 Generating AI summaries for all clusters...")
print("This may take a few minutes...")

count = cluster_summarizer.summarize_all_clusters()

print(f"✅ Generated {count} summaries!")
print("🎉 Your dashboard should now show cluster summaries!") 