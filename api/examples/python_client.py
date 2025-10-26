"""
Python Client Examples for AdventureWorks Analytics API

Examples of how to integrate the API into your applications:
- Churn prediction for CRM integration
- Return risk assessment for inventory systems
- Batch predictions for data processing
- Real-time alerts subscription

Author: AdventureWorks Data Science Team
Version: 1.0
Date: October 24, 2025
"""

import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

class AdventureWorksAPI:
    """
    Python client for AdventureWorks Analytics API

    Usage:
        api = AdventureWorksAPI(api_key="your-api-key-here")
        result = api.predict_churn(customer_id=12345, recency=90, frequency=10, monetary=5000)
        print(result)
    """

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "dev_key_12345"):
        """
        Initialize API client

        Args:
            base_url: API base URL (default: http://localhost:8000)
            api_key: Your API key
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """Make API request"""
        url = f"{self.base_url}{endpoint}"

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            raise

    def health_check(self) -> Dict:
        """Check API health"""
        return self._make_request("GET", "/api/v1/health")

    def get_models_status(self) -> List[Dict]:
        """Get status of all models"""
        return self._make_request("GET", "/api/v1/models/status")

    def predict_churn(
        self,
        recency: float,
        frequency: int,
        monetary: float,
        customer_id: Optional[int] = None,
        avg_order_value: Optional[float] = None,
        annual_income: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """
        Predict customer churn probability

        Args:
            recency: Days since last purchase
            frequency: Total number of purchases
            monetary: Total revenue from customer
            customer_id: Optional customer ID
            avg_order_value: Optional average order value
            annual_income: Optional annual income
            **kwargs: Additional customer features

        Returns:
            Dict with churn_probability, risk_level, recommendation
        """
        data = {
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "customer_id": customer_id,
            "avg_order_value": avg_order_value,
            "annual_income": annual_income,
            **kwargs
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        return self._make_request("POST", "/api/v1/predict/churn", data=data)

    def predict_return_risk(
        self,
        category: str,
        subcategory: str,
        list_price: float,
        product_id: Optional[int] = None,
        product_name: Optional[str] = None,
        return_rate: Optional[float] = None,
        sales_volume: Optional[int] = None
    ) -> Dict:
        """
        Predict product return risk

        Args:
            category: Product category (Bikes, Clothing, Accessories)
            subcategory: Product subcategory
            list_price: Product list price
            product_id: Optional product ID
            product_name: Optional product name
            return_rate: Optional historical return rate
            sales_volume: Optional sales volume

        Returns:
            Dict with risk_probability, risk_level, recommendation
        """
        data = {
            "category": category,
            "subcategory": subcategory,
            "list_price": list_price,
            "product_id": product_id,
            "product_name": product_name,
            "return_rate": return_rate,
            "sales_volume": sales_volume
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        return self._make_request("POST", "/api/v1/predict/return-risk", data=data)

    def get_customer_risk(self, customer_id: int) -> Dict:
        """
        Get churn risk for specific customer (requires customer data)

        Args:
            customer_id: Customer ID to look up

        Returns:
            Dict with churn prediction
        """
        return self._make_request("GET", f"/api/v1/customers/{customer_id}/risk")

    def batch_predict(self, predictions: List[Dict], prediction_type: str) -> Dict:
        """
        Make batch predictions (up to 1000 records)

        Args:
            predictions: List of prediction requests
            prediction_type: "churn" or "return_risk"

        Returns:
            Dict with results array and total_processed count
        """
        data = {
            "predictions": predictions,
            "prediction_type": prediction_type
        }

        return self._make_request("POST", "/api/v1/predict/batch", data=data)

    def subscribe_to_alerts(
        self,
        email: str,
        alert_types: List[str],
        threshold: float = 0.7,
        webhook_url: Optional[str] = None
    ) -> Dict:
        """
        Subscribe to real-time alerts

        Args:
            email: Email address for alerts
            alert_types: List of alert types ["churn", "return_risk", "revenue_drop"]
            threshold: Alert threshold (0-1)
            webhook_url: Optional webhook URL for notifications

        Returns:
            Dict with subscription_id and status
        """
        data = {
            "email": email,
            "alert_types": alert_types,
            "threshold": threshold,
            "webhook_url": webhook_url
        }

        return self._make_request("POST", "/api/v1/alerts/subscribe", data=data)


# ============================================================================
# EXAMPLE USE CASES
# ============================================================================

def example_1_single_churn_prediction():
    """Example 1: Predict churn for a single customer"""
    print("=" * 60)
    print("EXAMPLE 1: Single Churn Prediction")
    print("=" * 60)

    api = AdventureWorksAPI(api_key="dev_key_12345")

    # Predict churn for customer
    result = api.predict_churn(
        customer_id=12345,
        recency=90,  # 90 days since last purchase
        frequency=15,  # 15 total orders
        monetary=5000,  # $5,000 lifetime value
        avg_order_value=333.33,
        annual_income=75000
    )

    print(f"\nCustomer ID: {result['customer_id']}")
    print(f"Churn Probability: {result['churn_probability'] * 100:.1f}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Model Used: {result['model_used']}")
    print()


def example_2_return_risk_prediction():
    """Example 2: Predict return risk for a product"""
    print("=" * 60)
    print("EXAMPLE 2: Return Risk Prediction")
    print("=" * 60)

    api = AdventureWorksAPI(api_key="dev_key_12345")

    # Predict return risk
    result = api.predict_return_risk(
        product_id=680,
        product_name="Mountain-100 Silver, 44",
        category="Bikes",
        subcategory="Mountain Bikes",
        list_price=3399.99,
        return_rate=0.05,
        sales_volume=150
    )

    print(f"\nProduct: {result['product_name']} (ID: {result['product_id']})")
    print(f"Return Risk: {result['risk_probability'] * 100:.1f}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    print()


def example_3_batch_churn_prediction():
    """Example 3: Batch churn predictions"""
    print("=" * 60)
    print("EXAMPLE 3: Batch Churn Predictions")
    print("=" * 60)

    api = AdventureWorksAPI(api_key="dev_key_12345")

    # Prepare batch data
    customers = [
        {"customer_id": 101, "recency": 30, "frequency": 10, "monetary": 2000},
        {"customer_id": 102, "recency": 120, "frequency": 5, "monetary": 500},
        {"customer_id": 103, "recency": 60, "frequency": 20, "monetary": 8000},
        {"customer_id": 104, "recency": 180, "frequency": 3, "monetary": 300},
        {"customer_id": 105, "recency": 15, "frequency": 25, "monetary": 12000},
    ]

    # Make batch prediction
    result = api.batch_predict(
        predictions=customers,
        prediction_type="churn"
    )

    print(f"\nProcessed {result['total_processed']} customers")
    print(f"Timestamp: {result['timestamp']}\n")

    # Display results
    print(f"{'Customer ID':<15} {'Churn Risk':<12} {'Level':<10} {'Action':<50}")
    print("-" * 90)

    for pred in result['results']:
        print(f"{pred['customer_id']:<15} {pred['churn_probability']*100:>6.1f}%     {pred['risk_level']:<10} {pred['recommendation'][:47]}")

    print()


def example_4_crm_integration():
    """Example 4: CRM Integration - Daily high-risk customer report"""
    print("=" * 60)
    print("EXAMPLE 4: CRM Integration - Daily High-Risk Report")
    print("=" * 60)

    api = AdventureWorksAPI(api_key="prod_key_67890")

    # Simulate loading customer data from CRM
    crm_customers = pd.DataFrame({
        'customer_id': [201, 202, 203, 204, 205],
        'recency': [150, 30, 90, 200, 15],
        'frequency': [5, 20, 10, 3, 30],
        'monetary': [800, 6000, 3000, 400, 15000]
    })

    high_risk_customers = []

    print("\nScanning CRM database for high-risk customers...\n")

    for _, customer in crm_customers.iterrows():
        result = api.predict_churn(
            customer_id=int(customer['customer_id']),
            recency=float(customer['recency']),
            frequency=int(customer['frequency']),
            monetary=float(customer['monetary'])
        )

        if result['risk_level'] == 'High':
            high_risk_customers.append({
                'customer_id': result['customer_id'],
                'churn_probability': result['churn_probability'],
                'recommendation': result['recommendation']
            })

    print(f"Found {len(high_risk_customers)} high-risk customers:\n")

    for customer in high_risk_customers:
        print(f"Customer #{customer['customer_id']}:")
        print(f"  Risk: {customer['churn_probability']*100:.1f}%")
        print(f"  Action: {customer['recommendation']}")
        print()

    # In production: Send to CRM system via API or export to CSV
    if high_risk_customers:
        df = pd.DataFrame(high_risk_customers)
        # df.to_csv('high_risk_customers_daily.csv', index=False)
        print("✓ Results would be exported to CRM system")


def example_5_alert_subscription():
    """Example 5: Subscribe to real-time alerts"""
    print("=" * 60)
    print("EXAMPLE 5: Alert Subscription")
    print("=" * 60)

    api = AdventureWorksAPI(api_key="dev_key_12345")

    # Subscribe to alerts
    result = api.subscribe_to_alerts(
        email="sales-manager@adventureworks.com",
        alert_types=["churn", "return_risk"],
        threshold=0.7,  # Alert when risk >= 70%
        webhook_url="https://your-crm.com/webhooks/analytics"
    )

    print(f"\nSubscription ID: {result['subscription_id']}")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print()


def example_6_model_monitoring():
    """Example 6: Monitor API and model status"""
    print("=" * 60)
    print("EXAMPLE 6: API & Model Status Monitoring")
    print("=" * 60)

    api = AdventureWorksAPI(api_key="dev_key_12345")

    # Check API health
    health = api.health_check()
    print(f"\nAPI Status: {health['status']}")
    print(f"Version: {health['version']}")
    print(f"Models Loaded: {health['models_loaded']}")
    print(f"Timestamp: {health['timestamp']}")

    # Check model status
    models = api.get_models_status()
    print(f"\n{'Model':<25} {'Type':<25} {'Accuracy':<12} {'Status':<10}")
    print("-" * 75)

    for model in models:
        accuracy_str = f"{model['accuracy']:.2f}%" if model['accuracy'] else "N/A"
        print(f"{model['model_name']:<25} {model['model_type']:<25} {accuracy_str:<12} {model['status']:<10}")

    print()


def example_7_webhook_integration():
    """Example 7: Webhook integration for external systems"""
    print("=" * 60)
    print("EXAMPLE 7: Webhook Integration Example")
    print("=" * 60)

    api = AdventureWorksAPI(api_key="dev_key_12345")

    # Make prediction
    result = api.predict_churn(
        customer_id=999,
        recency=150,
        frequency=5,
        monetary=1000
    )

    # Simulate sending to webhook
    webhook_payload = {
        "event": "high_churn_risk",
        "timestamp": datetime.now().isoformat(),
        "customer_id": result['customer_id'],
        "churn_probability": result['churn_probability'],
        "risk_level": result['risk_level'],
        "recommendation": result['recommendation']
    }

    print("\nWebhook payload that would be sent to CRM:")
    print(json.dumps(webhook_payload, indent=2))
    print()

    # In production:
    # requests.post("https://your-crm.com/webhooks/churn", json=webhook_payload)


# ============================================================================
# MAIN - RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Run all examples

    Prerequisites:
    1. Start the API server: uvicorn api.main:app --reload
    2. Ensure models are trained and saved
    3. Update API key if needed
    """

    print("\n" + "=" * 60)
    print("ADVENTUREWORKS ANALYTICS API - PYTHON CLIENT EXAMPLES")
    print("=" * 60 + "\n")

    try:
        # Run examples
        example_1_single_churn_prediction()
        example_2_return_risk_prediction()
        example_3_batch_churn_prediction()
        example_4_crm_integration()
        example_5_alert_subscription()
        example_6_model_monitoring()
        example_7_webhook_integration()

        print("=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        print("\nMake sure the API server is running:")
        print("  uvicorn api.main:app --reload --port 8000")
        print()
