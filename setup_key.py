#!/usr/bin/env python3
"""
Simple script to set up your OpenAI API key
"""

import os

def setup_openai_key():
    print("🔑 OpenAI API Key Setup")
    print("=" * 40)
    
    # Get the key from user
    api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No key provided. Exiting.")
        return False
    
    if not api_key.startswith("sk-"):
        print("⚠️  Warning: OpenAI keys usually start with 'sk-'")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            return False
    
    # Update config.py
    try:
        with open('config.py', 'r') as f:
            content = f.read()
        
        # Replace the placeholder with actual key
        new_content = content.replace(
            'OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-actual-openai-key-here")',
            f'OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "{api_key}")'
        )
        
        with open('config.py', 'w') as f:
            f.write(new_content)
        
        print("✅ API key saved to config.py")
        
        # Set environment variable for current session
        os.environ['OPENAI_API_KEY'] = api_key
        print("✅ API key set for current session")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

def test_key():
    """Test the API key"""
    print("\n🧪 Testing API key...")
    
    try:
        # Try to import and test
        from config import config
        
        if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "sk-your-actual-openai-key-here":
            print("✅ API key loaded from config")
            print(f"Key: {config.OPENAI_API_KEY[:10]}...{config.OPENAI_API_KEY[-4:]}")
            
            # Test with OpenAI (if available)
            try:
                import openai
                openai.api_key = config.OPENAI_API_KEY
                
                # Simple test call
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                print("✅ API key works! OpenAI responded successfully.")
                return True
                
            except ImportError:
                print("⚠️  OpenAI library not installed, but key is configured")
                print("   Install with: pip install openai")
                return True
                
            except Exception as e:
                print(f"❌ API key test failed: {e}")
                return False
        else:
            print("❌ No valid API key found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing key: {e}")
        return False

def main():
    """Main setup function"""
    if setup_openai_key():
        test_key()
        
        print("\n🚀 Next Steps:")
        print("1. Run full analysis: python main.py")
        print("2. View results: http://localhost:8080")
        print("3. Try embeddings and clustering!")
    else:
        print("❌ Setup failed. Try again.")

if __name__ == "__main__":
    main() 