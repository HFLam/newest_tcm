#!/usr/bin/env python3
"""
Test script to verify health check endpoints work correctly
"""

import requests
import time
import json

def test_health_endpoints(base_url="http://localhost:8000"):
    """Test all health check endpoints"""
    print("🏥 Testing Health Check Endpoints")
    print("=" * 50)
    
    endpoints_to_test = [
        ("/", "Root endpoint"),
        ("/ping", "Ping endpoint"),
        ("/health", "Health check endpoint"),
        ("/ready", "Readiness check endpoint")
    ]
    
    for endpoint, description in endpoints_to_test:
        print(f"\n🔍 Testing {description}: {base_url}{endpoint}")
        
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}{endpoint}", timeout=30)
            response_time = (time.time() - start_time) * 1000
            
            print(f"   ✅ Status Code: {response.status_code}")
            print(f"   ⏱️  Response Time: {response_time:.2f}ms")
            
            # Try to parse JSON response
            try:
                json_response = response.json()
                print(f"   📄 Response: {json.dumps(json_response, indent=2)}")
            except:
                # For non-JSON responses like /ping
                print(f"   📄 Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection failed - server not running?")
        except requests.exceptions.Timeout:
            print(f"   ⏰ Request timed out (>30s)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 Health check testing complete!")
    print(f"ℹ️  If all endpoints return 200 status codes, Railway health checks should pass.")

if __name__ == "__main__":
    # Test with default localhost
    test_health_endpoints()
    
    print(f"\n" + "="*50)
    print(f"🚀 To test with a live Railway deployment:")
    print(f"   python3 test_health_check.py")
    print(f"   # Then manually change the base_url in the script to your Railway URL") 