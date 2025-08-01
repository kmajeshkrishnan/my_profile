#!/usr/bin/env python3
"""
Test script for RAG system functionality
"""

import asyncio
import aiohttp
import json
import time

async def test_rag_system():
    """Test the RAG system with various questions."""
    
    base_url = "http://localhost:8000"
    
    # Test questions
    test_questions = [
        "What is your name?",
        "What is your educational background?",
        "What are your technical skills?",
        "Tell me about your work experience",
        "What projects have you worked on?",
        "What are your areas of expertise?",
        "What technologies do you use?",
        "What is your experience with machine learning?"
    ]
    
    print("🧪 Testing RAG System")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        # Test health check
        print("\n1. Testing health check...")
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ Health check passed: {health_data['status']}")
                    if 'rag_health' in health_data:
                        print(f"   RAG Status: {health_data['rag_health']['status']}")
                else:
                    print(f"❌ Health check failed: {response.status}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        # Test RAG info
        print("\n2. Testing RAG info...")
        try:
            async with session.get(f"{base_url}/rag/info") as response:
                if response.status == 200:
                    rag_info = await response.json()
                    print(f"✅ RAG info retrieved:")
                    print(f"   Initialized: {rag_info['initialized']}")
                    print(f"   Total Queries: {rag_info['total_queries']}")
                    print(f"   Success Rate: {rag_info['success_rate']:.2%}")
                else:
                    print(f"❌ RAG info failed: {response.status}")
        except Exception as e:
            print(f"❌ RAG info error: {e}")
        
        # Test RAG queries
        print("\n3. Testing RAG queries...")
        for i, question in enumerate(test_questions, 1):
            print(f"\n   Question {i}: {question}")
            try:
                payload = {
                    "query": question,
                    "include_metadata": True
                }
                
                async with session.post(
                    f"{base_url}/rag/query",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result['success']:
                            print(f"   ✅ Response: {result['response'][:100]}...")
                            print(f"   ⏱️  Processing time: {result['processing_time']:.3f}s")
                            if result.get('metadata'):
                                print(f"   📊 Success rate: {result['metadata']['success_rate']:.2%}")
                        else:
                            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
                    else:
                        print(f"   ❌ HTTP Error: {response.status}")
                        
            except Exception as e:
                print(f"   ❌ Request error: {e}")
            
            # Small delay between requests
            await asyncio.sleep(1)
    
    print("\n" + "=" * 50)
    print("🎉 RAG System Test Complete!")

if __name__ == "__main__":
    print("Starting RAG system test...")
    print("Make sure the backend server is running on http://localhost:8000")
    print()
    
    try:
        asyncio.run(test_rag_system())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}") 