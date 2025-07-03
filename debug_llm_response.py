#!/usr/bin/env python3
"""
Debug script to see what the LLM is actually returning
"""
from summarization import cluster_summarizer
from database import get_db_session
from models import Cluster

def debug_llm_response():
    cluster_summarizer.openai_client = cluster_summarizer._setup_openai_client()
    
    # Get a cluster
    db = get_db_session()
    try:
        cluster = db.query(Cluster).filter(Cluster.is_noise.is_(False)).first()
        if cluster:
            print(f"Testing cluster {cluster.id}")
            cluster_data = cluster_summarizer.get_cluster_data(cluster.id)
            prompt = cluster_summarizer.create_gpt_prompt(cluster_data)
            
            print("=== PROMPT ===")
            print(prompt[:500] + "...")
            print("\n=== RAW LLM RESPONSE ===")
            
            # Call LLM directly
            import openai
            from config import config
            
            model_name = config.AZURE_OPENAI_DEPLOYMENT_NAME if config.USE_AZURE_OPENAI else config.OPENAI_MODEL
            response = cluster_summarizer.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            
            raw_response = response.choices[0].message.content
            print(raw_response)
            print("\n=== PARSED RESPONSE ===")
            
            parsed = cluster_summarizer.generate_gpt_summary(prompt)
            print(parsed)
    finally:
        db.close()

if __name__ == "__main__":
    debug_llm_response()
