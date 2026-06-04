#!/usr/bin/env python3
"""
Health Check & Connection Test Script for Grey Backend

Tests:
1. Backend API connectivity
2. Database connection
3. Route availability
4. CORS configuration

Usage:
  python health_check.py [--host localhost] [--port 8000] [--db-port 27017]
"""

import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import time

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    import httpx
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Install missing packages: pip install httpx pymongo")
    sys.exit(1)


class HealthCheck:
    def __init__(self, backend_host="localhost", backend_port=8000, mongo_uri="mongodb://localhost:27017"):
        self.backend_url = f"http://{backend_host}:{backend_port}"
        self.mongo_uri = mongo_uri
        self.results: Dict[str, Any] = {}
        self.passed = 0
        self.failed = 0

    def log_test(self, name: str, status: bool, message: str = ""):
        """Log test result"""
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
        if message:
            print(f"   └─ {message}")
        
        self.results[name] = {"status": status, "message": message}
        if status:
            self.passed += 1
        else:
            self.failed += 1

    def test_backend_root(self):
        """Test backend root endpoint"""
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.backend_url}/")
                status = response.status_code == 200
                data = response.json() if status else {}
                
                self.log_test(
                    "Backend Root Endpoint",
                    status,
                    f"Status: {response.status_code}, Service: {data.get('service', 'unknown')}"
                )
        except Exception as e:
            self.log_test("Backend Root Endpoint", False, str(e))

    def test_backend_health(self):
        """Test backend health endpoint"""
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.backend_url}/health")
                status = response.status_code == 200
                
                self.log_test(
                    "Backend Health Endpoint",
                    status,
                    f"Status: {response.status_code}"
                )
        except Exception as e:
            self.log_test("Backend Health Endpoint", False, str(e))

    def test_backend_db_health(self):
        """Test backend database health endpoint"""
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.backend_url}/health/db")
                data = response.json()
                status = response.status_code == 200 and data.get("status") == "healthy"
                
                self.log_test(
                    "Backend DB Health",
                    status,
                    f"Status: {data.get('status', 'unknown')}, Database: {data.get('database', 'unknown')}"
                )
        except Exception as e:
            self.log_test("Backend DB Health", False, str(e))

    def test_campaigns_endpoint(self):
        """Test campaigns list endpoint"""
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.backend_url}/api/campaigns?type=backtest&limit=1")
                status = response.status_code == 200
                data = response.json()
                
                count = len(data) if isinstance(data, list) else "unknown"
                self.log_test(
                    "Campaigns List Endpoint",
                    status,
                    f"Status: {response.status_code}, Records: {count}"
                )
        except Exception as e:
            self.log_test("Campaigns List Endpoint", False, str(e))

    def test_mongodb_direct(self):
        """Test MongoDB connection directly"""
        try:
            client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            # Test ping
            client.admin.command('ping')
            
            # Get server info
            server_info = client.server_info()
            version = server_info.get('version', 'unknown')
            
            self.log_test(
                "MongoDB Direct Connection",
                True,
                f"Connected! Version: {version}"
            )
            client.close()
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.log_test("MongoDB Direct Connection", False, f"Connection timeout: {str(e)}")
        except Exception as e:
            self.log_test("MongoDB Direct Connection", False, str(e))

    def test_cors_headers(self):
        """Test CORS headers"""
        try:
            with httpx.Client(timeout=10) as client:
                response = client.options(
                    f"{self.backend_url}/api/campaigns",
                    headers={"Origin": "http://localhost:5721"}
                )
                
                cors_header = response.headers.get("access-control-allow-origin")
                has_cors = bool(cors_header)
                
                self.log_test(
                    "CORS Headers",
                    has_cors,
                    f"Allow-Origin: {cors_header or 'NOT SET'}"
                )
        except Exception as e:
            self.log_test("CORS Headers", False, str(e))

    async def run_async_tests(self):
        """Run async tests"""
        # Can add async tests here if needed
        pass

    def run_all(self):
        """Run all health checks"""
        print("=" * 60)
        print("GREY BACKEND HEALTH CHECK")
        print("=" * 60)
        print(f"Backend URL: {self.backend_url}")
        print(f"MongoDB URI: {self.mongo_uri}")
        print("=" * 60)
        print()

        # Run tests
        self.test_backend_root()
        self.test_backend_health()
        self.test_backend_db_health()
        self.test_mongodb_direct()
        self.test_campaigns_endpoint()
        self.test_cors_headers()

        print()
        print("=" * 60)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        print("=" * 60)

        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description="Grey Backend Health Check")
    parser.add_argument("--host", default="localhost", help="Backend host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017", help="MongoDB URI")
    args = parser.parse_args()

    checker = HealthCheck(
        backend_host=args.host,
        backend_port=args.port,
        mongo_uri=args.mongo_uri
    )

    success = checker.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
