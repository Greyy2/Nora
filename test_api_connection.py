#!/usr/bin/env python3
"""
Test Frontend-Backend API Connection

This script tests various API endpoints to verify frontend-backend communication works.

Usage:
  python test_api_connection.py [--backend http://localhost:8000]
"""

import sys
import time
import json
import argparse
from typing import Dict, Any
import urllib.request
import urllib.error

class APITester:
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url.rstrip('/')
        self.passed = 0
        self.failed = 0
        self.results: Dict[str, Any] = {}

    def log_test(self, name: str, status: bool, message: str = "", details: Dict[str, Any] = None):
        """Log test result"""
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
        if message:
            print(f"   └─ {message}")
        
        result = {"status": status, "message": message}
        if details:
            result.update(details)
        
        self.results[name] = result
        if status:
            self.passed += 1
        else:
            self.failed += 1

    def test_get(self, endpoint: str, expected_status: int = 200) -> tuple[bool, Dict]:
        """Test GET request"""
        try:
            url = f"{self.backend_url}{endpoint}"
            req = urllib.request.Request(url, method='GET')
            req.add_header('Accept', 'application/json')
            
            with urllib.request.urlopen(req, timeout=5) as response:
                status = response.status
                body = json.loads(response.read().decode())
                
                if status == expected_status:
                    return True, {"status": status, "data": body}
                else:
                    return False, {"status": status, "error": f"Expected {expected_status}, got {status}"}
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode())
            except:
                body = {"error": str(e)}
            return False, {"status": e.code, "error": str(e), "body": body}
        except Exception as e:
            return False, {"error": str(e), "error_type": type(e).__name__}

    def test_post(self, endpoint: str, payload: Dict = None, expected_status: int = 200) -> tuple[bool, Dict]:
        """Test POST request"""
        try:
            url = f"{self.backend_url}{endpoint}"
            data = json.dumps(payload or {}).encode('utf-8')
            req = urllib.request.Request(url, data=data, method='POST')
            req.add_header('Content-Type', 'application/json')
            req.add_header('Accept', 'application/json')
            
            with urllib.request.urlopen(req, timeout=5) as response:
                status = response.status
                body = json.loads(response.read().decode())
                
                if status == expected_status:
                    return True, {"status": status, "data": body}
                else:
                    return False, {"status": status, "error": f"Expected {expected_status}, got {status}"}
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode())
            except:
                body = {"error": str(e)}
            return False, {"status": e.code, "error": str(e), "body": body}
        except Exception as e:
            return False, {"error": str(e), "error_type": type(e).__name__}

    def run_tests(self):
        """Run all API tests"""
        print("=" * 70)
        print("GREY BACKEND - API CONNECTION TEST")
        print("=" * 70)
        print(f"Backend URL: {self.backend_url}")
        print("=" * 70)
        print()

        # Test 1: Root endpoint
        print("📋 Testing Basic Endpoints...")
        status, result = self.test_get("/")
        self.log_test(
            "GET / (Root)",
            status,
            f"Status: {result.get('status')}",
            result
        )

        # Test 2: Health endpoint
        status, result = self.test_get("/health")
        self.log_test(
            "GET /health",
            status,
            f"Status: {result.get('status')}",
            result
        )

        # Test 3: DB Health endpoint
        status, result = self.test_get("/health/db", expected_status=200)
        self.log_test(
            "GET /health/db",
            status,
            f"Status: {result.get('status')}, Database: {result.get('data', {}).get('database', 'unknown')}",
            result
        )

        print()
        print("🔄 Testing Campaign API Endpoints...")

        # Test 4: List campaigns (backtest)
        status, result = self.test_get("/api/campaigns?type=backtest&limit=5")
        campaign_count = len(result.get('data', [])) if isinstance(result.get('data'), list) else 0
        self.log_test(
            "GET /api/campaigns (backtest)",
            status,
            f"Status: {result.get('status')}, Count: {campaign_count}",
            result
        )

        # Test 5: List campaigns (WFA)
        status, result = self.test_get("/api/campaigns?type=wfa&limit=5")
        campaign_count = len(result.get('data', [])) if isinstance(result.get('data'), list) else 0
        self.log_test(
            "GET /api/campaigns (wfa)",
            status,
            f"Status: {result.get('status')}, Count: {campaign_count}",
            result
        )

        print()
        print("📊 Testing Data Endpoints...")

        # Test 6: List data
        status, result = self.test_get("/api/data")
        self.log_test(
            "GET /api/data",
            status,
            f"Status: {result.get('status')}",
            result
        )

        # Test 7: List chart types
        status, result = self.test_get("/api/chart/types")
        self.log_test(
            "GET /api/chart/types",
            status,
            f"Status: {result.get('status')}",
            result
        )

        print()
        print("=" * 70)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        print()

        if self.failed == 0:
            print("✅ All tests passed! Frontend-Backend connection is working.")
        else:
            print(f"❌ {self.failed} test(s) failed. Check the errors above.")
            print("\nCommon issues:")
            print("  1. Backend is not running (start with: uvicorn backend.main:app --reload)")
            print("  2. MongoDB is not running (start with: docker run -d -p 27020:27017 mongo:6.0)")
            print("  3. Backend port is different (use --backend http://localhost:8001)")
            print("  4. CORS is not configured (check CORS_ORIGINS in backend/core/config.py)")

        return self.failed == 0

    def print_summary(self):
        """Print test summary in JSON"""
        print("\nTest Summary (JSON):")
        print(json.dumps({
            "total": self.passed + self.failed,
            "passed": self.passed,
            "failed": self.failed,
            "results": self.results
        }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Test Grey Backend API Connection")
    parser.add_argument(
        "--backend",
        default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results in JSON format"
    )
    args = parser.parse_args()

    tester = APITester(backend_url=args.backend)
    success = tester.run_tests()
    
    if args.json:
        tester.print_summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
