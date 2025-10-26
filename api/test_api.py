"""
API Testing Script
Phase 7: Real-Time API & Integration

Comprehensive tests for all API endpoints:
- Health checks
- Churn prediction
- Return risk prediction
- Batch predictions
- Alert subscriptions
- Error handling

Author: AdventureWorks Data Science Team
Version: 1.0
Date: October 24, 2025
"""

import requests
import json
import time
from typing import Dict, List
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "dev_key_12345"

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_test_header(test_name: str):
    """Print formatted test header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}TEST: {test_name}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")

def make_request(method: str, endpoint: str, data: Dict = None) -> Dict:
    """Make API request with error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return {"success": True, "data": response.json(), "status": response.status_code}

    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": str(e), "status": e.response.status_code, "detail": e.response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# TEST CASES
# ============================================================================

def test_1_root_endpoint():
    """Test 1: Root endpoint"""
    print_test_header("Root Endpoint")

    result = make_request("GET", "/")

    if result["success"]:
        data = result["data"]
        print_success(f"Root endpoint accessible")
        print_info(f"Message: {data.get('message')}")
        print_info(f"Version: {data.get('version')}")
        print_info(f"Status: {data.get('status')}")
        return True
    else:
        print_error(f"Failed: {result.get('error')}")
        return False

def test_2_health_check():
    """Test 2: Health check endpoint"""
    print_test_header("Health Check")

    result = make_request("GET", "/api/v1/health")

    if result["success"]:
        data = result["data"]
        print_success(f"API is {data.get('status')}")
        print_info(f"Version: {data.get('version')}")
        print_info(f"Models loaded: {data.get('models_loaded')}")
        print_info(f"Timestamp: {data.get('timestamp')}")
        return True
    else:
        print_error(f"Health check failed: {result.get('error')}")
        return False

def test_3_models_status():
    """Test 3: Models status endpoint"""
    print_test_header("Models Status")

    result = make_request("GET", "/api/v1/models/status")

    if result["success"]:
        models = result["data"]
        print_success(f"Retrieved {len(models)} models")

        for model in models:
            print_info(f"\nModel: {model['model_name']}")
            print(f"  Type: {model['model_type']}")
            print(f"  Accuracy: {model['accuracy']}%")
            print(f"  Features: {model['features_count']}")
            print(f"  Status: {model['status']}")

        return True
    else:
        print_error(f"Failed: {result.get('error')}")
        return False

def test_4_churn_prediction_low_risk():
    """Test 4: Churn prediction - Low risk customer"""
    print_test_header("Churn Prediction - Low Risk")

    payload = {
        "customer_id": 101,
        "recency": 15,  # Recent purchase
        "frequency": 25,  # Many orders
        "monetary": 10000,  # High value
        "avg_order_value": 400,
        "annual_income": 80000
    }

    print_info(f"Testing customer: ID={payload['customer_id']}, Recency={payload['recency']} days")

    result = make_request("POST", "/api/v1/predict/churn", payload)

    if result["success"]:
        data = result["data"]
        print_success(f"Prediction successful")
        print_info(f"Churn Probability: {data['churn_probability']*100:.1f}%")
        print_info(f"Risk Level: {data['risk_level']}")
        print_info(f"Recommendation: {data['recommendation']}")
        print_info(f"Model: {data['model_used']}")

        # Validate low risk
        if data['risk_level'] == 'Low':
            print_success("✓ Correctly classified as Low risk")
        return True
    else:
        print_error(f"Prediction failed: {result.get('error')}")
        print_error(f"Detail: {result.get('detail')}")
        return False

def test_5_churn_prediction_high_risk():
    """Test 5: Churn prediction - High risk customer"""
    print_test_header("Churn Prediction - High Risk")

    payload = {
        "customer_id": 999,
        "recency": 180,  # 6 months inactive
        "frequency": 3,  # Few orders
        "monetary": 500,  # Low value
        "avg_order_value": 166,
        "annual_income": 50000
    }

    print_info(f"Testing customer: ID={payload['customer_id']}, Recency={payload['recency']} days")

    result = make_request("POST", "/api/v1/predict/churn", payload)

    if result["success"]:
        data = result["data"]
        print_success(f"Prediction successful")
        print_info(f"Churn Probability: {data['churn_probability']*100:.1f}%")
        print_info(f"Risk Level: {data['risk_level']}")
        print_info(f"Recommendation: {data['recommendation']}")

        # Validate high risk
        if data['risk_level'] in ['High', 'Medium']:
            print_success("✓ Correctly flagged as at-risk")
        return True
    else:
        print_error(f"Prediction failed: {result.get('error')}")
        return False

def test_6_return_risk_prediction_bikes():
    """Test 6: Return risk prediction - Bikes (high risk)"""
    print_test_header("Return Risk Prediction - Bikes")

    payload = {
        "product_id": 680,
        "product_name": "Mountain-100 Silver, 44",
        "category": "Bikes",
        "subcategory": "Mountain Bikes",
        "list_price": 3399.99,
        "return_rate": 0.05,
        "sales_volume": 150
    }

    print_info(f"Testing product: {payload['product_name']}")

    result = make_request("POST", "/api/v1/predict/return-risk", payload)

    if result["success"]:
        data = result["data"]
        print_success(f"Prediction successful")
        print_info(f"Return Risk: {data['risk_probability']*100:.1f}%")
        print_info(f"Risk Level: {data['risk_level']}")
        print_info(f"Recommendation: {data['recommendation']}")
        print_info(f"Model: {data['model_used']}")
        return True
    else:
        print_error(f"Prediction failed: {result.get('error')}")
        return False

def test_7_return_risk_prediction_accessories():
    """Test 7: Return risk prediction - Accessories (low risk)"""
    print_test_header("Return Risk Prediction - Accessories")

    payload = {
        "product_id": 200,
        "product_name": "Water Bottle",
        "category": "Accessories",
        "subcategory": "Bottles and Cages",
        "list_price": 12.99,
        "return_rate": 0.01,
        "sales_volume": 500
    }

    print_info(f"Testing product: {payload['product_name']}")

    result = make_request("POST", "/api/v1/predict/return-risk", payload)

    if result["success"]:
        data = result["data"]
        print_success(f"Prediction successful")
        print_info(f"Return Risk: {data['risk_probability']*100:.1f}%")
        print_info(f"Risk Level: {data['risk_level']}")

        if data['risk_level'] == 'Low':
            print_success("✓ Correctly classified as Low risk")
        return True
    else:
        print_error(f"Prediction failed: {result.get('error')}")
        return False

def test_8_batch_predictions():
    """Test 8: Batch predictions"""
    print_test_header("Batch Predictions")

    payload = {
        "prediction_type": "churn",
        "predictions": [
            {"customer_id": 201, "recency": 30, "frequency": 10, "monetary": 2000},
            {"customer_id": 202, "recency": 120, "frequency": 5, "monetary": 500},
            {"customer_id": 203, "recency": 60, "frequency": 15, "monetary": 5000},
        ]
    }

    print_info(f"Processing batch of {len(payload['predictions'])} predictions")

    result = make_request("POST", "/api/v1/predict/batch", payload)

    if result["success"]:
        data = result["data"]
        print_success(f"Batch processed: {data['total_processed']} predictions")

        for i, pred in enumerate(data['results'], 1):
            print(f"\n  {i}. Customer {pred['customer_id']}:")
            print(f"     Risk: {pred['churn_probability']*100:.1f}% ({pred['risk_level']})")

        return True
    else:
        print_error(f"Batch prediction failed: {result.get('error')}")
        return False

def test_9_alert_subscription():
    """Test 9: Alert subscription"""
    print_test_header("Alert Subscription")

    payload = {
        "email": "test@adventureworks.com",
        "alert_types": ["churn", "return_risk"],
        "threshold": 0.7,
        "webhook_url": "https://example.com/webhook"
    }

    print_info(f"Subscribing {payload['email']} to alerts")

    result = make_request("POST", "/api/v1/alerts/subscribe", payload)

    if result["success"]:
        data = result["data"]
        print_success(f"Subscription created")
        print_info(f"Subscription ID: {data['subscription_id']}")
        print_info(f"Status: {data['status']}")
        print_info(f"Message: {data['message']}")
        return True
    else:
        print_error(f"Subscription failed: {result.get('error')}")
        return False

def test_10_authentication_invalid_key():
    """Test 10: Authentication with invalid API key"""
    print_test_header("Authentication - Invalid API Key")

    # Temporarily use invalid key
    global API_KEY
    original_key = API_KEY
    API_KEY = "invalid_key_12345"

    result = make_request("GET", "/api/v1/health")

    # Restore original key
    API_KEY = original_key

    if not result["success"] and result.get("status") == 403:
        print_success("✓ Correctly rejected invalid API key")
        print_info(f"Status: {result['status']} (Forbidden)")
        return True
    else:
        print_error("Security issue: Invalid key was accepted!")
        return False

def test_11_input_validation():
    """Test 11: Input validation"""
    print_test_header("Input Validation")

    # Test invalid category
    payload = {
        "category": "InvalidCategory",  # Invalid
        "subcategory": "Test",
        "list_price": 100
    }

    print_info("Testing with invalid category")

    result = make_request("POST", "/api/v1/predict/return-risk", payload)

    if not result["success"] and result.get("status") in [400, 422]:
        print_success("✓ Correctly rejected invalid input")
        print_info(f"Status: {result['status']}")
        return True
    else:
        print_error("Validation failed - invalid data accepted")
        return False

def test_12_performance():
    """Test 12: API performance"""
    print_test_header("Performance Test")

    payload = {
        "customer_id": 500,
        "recency": 90,
        "frequency": 10,
        "monetary": 5000
    }

    print_info("Testing response time (10 requests)")

    times = []
    for i in range(10):
        start = time.time()
        result = make_request("POST", "/api/v1/predict/churn", payload)
        elapsed = time.time() - start
        times.append(elapsed)

        if not result["success"]:
            print_error(f"Request {i+1} failed")
            return False

    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    print_success("Performance test completed")
    print_info(f"Average response time: {avg_time*1000:.0f}ms")
    print_info(f"Min: {min_time*1000:.0f}ms, Max: {max_time*1000:.0f}ms")

    if avg_time < 1.0:  # Under 1 second
        print_success("✓ Performance within acceptable range (<1s)")
        return True
    else:
        print_error(f"Performance issue: Average {avg_time:.2f}s")
        return False

# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all API tests"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("=" * 70)
    print(" ADVENTUREWORKS ANALYTICS API - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"{Colors.ENDC}\n")

    print_info(f"API Base URL: {API_BASE_URL}")
    print_info(f"API Key: {API_KEY[:10]}...")
    print_info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    tests = [
        test_1_root_endpoint,
        test_2_health_check,
        test_3_models_status,
        test_4_churn_prediction_low_risk,
        test_5_churn_prediction_high_risk,
        test_6_return_risk_prediction_bikes,
        test_7_return_risk_prediction_accessories,
        test_8_batch_predictions,
        test_9_alert_subscription,
        test_10_authentication_invalid_key,
        test_11_input_validation,
        test_12_performance,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print_error(f"Test crashed: {str(e)}")
            results.append((test_func.__name__, False))

    # Summary
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    print(f"{Colors.ENDC}\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = f"{Colors.OKGREEN}PASS{Colors.ENDC}" if result else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        print(f"{status}  {test_name}")

    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed ({passed/total*100:.0f}%){Colors.ENDC}\n")

    if passed == total:
        print(f"{Colors.OKGREEN}{Colors.BOLD}✅ ALL TESTS PASSED!{Colors.ENDC}\n")
        return True
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}❌ SOME TESTS FAILED{Colors.ENDC}\n")
        return False

if __name__ == "__main__":
    """
    Run test suite

    Prerequisites:
    1. Start API server: uvicorn api.main:app --reload --port 8000
    2. Ensure models are trained and available
    3. Run: python api/test_api.py
    """

    try:
        success = run_all_tests()
        exit(0 if success else 1)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Tests interrupted by user{Colors.ENDC}\n")
        exit(130)

    except Exception as e:
        print(f"\n{Colors.FAIL}Fatal error: {str(e)}{Colors.ENDC}\n")
        exit(1)
