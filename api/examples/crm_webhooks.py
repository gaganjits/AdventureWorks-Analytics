"""
CRM Webhook Integration Examples
Phase 7: Real-Time API & Integration

Webhook handlers for popular CRM systems:
- Salesforce
- HubSpot
- Dynamics 365
- Zendesk
- Custom webhooks

Author: AdventureWorks Data Science Team
Version: 1.0
Date: October 24, 2025
"""

import requests
import json
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
import hmac

# ============================================================================
# SALESFORCE INTEGRATION
# ============================================================================

class SalesforceIntegration:
    """
    Integrate churn predictions with Salesforce CRM

    Features:
    - Update customer churn score field
    - Create tasks for sales reps
    - Trigger workflows based on risk level
    """

    def __init__(self, instance_url: str, access_token: str):
        """
        Initialize Salesforce connection

        Args:
            instance_url: Your Salesforce instance URL (e.g., https://yourinstance.salesforce.com)
            access_token: OAuth access token
        """
        self.instance_url = instance_url.rstrip('/')
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

    def update_customer_churn_score(self, customer_email: str, churn_probability: float, risk_level: str) -> bool:
        """Update custom Churn_Score__c field in Salesforce Contact"""

        # Find contact by email
        query = f"SELECT Id FROM Contact WHERE Email = '{customer_email}' LIMIT 1"
        url = f"{self.instance_url}/services/data/v58.0/query"

        try:
            response = requests.get(url, headers=self.headers, params={'q': query})
            response.raise_for_status()

            records = response.json().get('records', [])
            if not records:
                print(f"Contact not found: {customer_email}")
                return False

            contact_id = records[0]['Id']

            # Update churn score
            update_url = f"{self.instance_url}/services/data/v58.0/sobjects/Contact/{contact_id}"
            update_data = {
                "Churn_Score__c": round(churn_probability * 100, 2),
                "Churn_Risk_Level__c": risk_level,
                "Last_Churn_Update__c": datetime.now().isoformat()
            }

            response = requests.patch(update_url, headers=self.headers, json=update_data)
            response.raise_for_status()

            print(f"✓ Updated Salesforce contact {contact_id}: {risk_level} risk ({churn_probability*100:.1f}%)")
            return True

        except Exception as e:
            print(f"❌ Salesforce update failed: {str(e)}")
            return False

    def create_retention_task(self, customer_email: str, recommendation: str) -> bool:
        """Create task for sales rep to follow up with at-risk customer"""

        # Find contact
        query = f"SELECT Id, OwnerId FROM Contact WHERE Email = '{customer_email}' LIMIT 1"
        url = f"{self.instance_url}/services/data/v58.0/query"

        try:
            response = requests.get(url, headers=self.headers, params={'q': query})
            response.raise_for_status()

            records = response.json().get('records', [])
            if not records:
                return False

            contact_id = records[0]['Id']
            owner_id = records[0]['OwnerId']

            # Create task
            task_url = f"{self.instance_url}/services/data/v58.0/sobjects/Task"
            task_data = {
                "WhoId": contact_id,
                "OwnerId": owner_id,
                "Subject": "High Churn Risk - Retention Action Required",
                "Description": f"Analytics Alert: {recommendation}",
                "Priority": "High",
                "Status": "Not Started",
                "ActivityDate": datetime.now().date().isoformat()
            }

            response = requests.post(task_url, headers=self.headers, json=task_data)
            response.raise_for_status()

            print(f"✓ Created Salesforce task for contact {contact_id}")
            return True

        except Exception as e:
            print(f"❌ Task creation failed: {str(e)}")
            return False


# ============================================================================
# HUBSPOT INTEGRATION
# ============================================================================

class HubSpotIntegration:
    """
    Integrate churn predictions with HubSpot CRM

    Features:
    - Update contact properties
    - Create engagement tasks
    - Trigger workflows
    """

    def __init__(self, api_key: str):
        """
        Initialize HubSpot connection

        Args:
            api_key: HubSpot API key
        """
        self.api_key = api_key
        self.base_url = "https://api.hubapi.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def update_contact_churn_score(self, email: str, churn_probability: float, risk_level: str) -> bool:
        """Update custom churn properties on HubSpot contact"""

        try:
            # Find contact by email
            search_url = f"{self.base_url}/crm/v3/objects/contacts/search"
            search_payload = {
                "filterGroups": [{
                    "filters": [{
                        "propertyName": "email",
                        "operator": "EQ",
                        "value": email
                    }]
                }]
            }

            response = requests.post(search_url, headers=self.headers, json=search_payload)
            response.raise_for_status()

            results = response.json().get('results', [])
            if not results:
                print(f"Contact not found: {email}")
                return False

            contact_id = results[0]['id']

            # Update properties
            update_url = f"{self.base_url}/crm/v3/objects/contacts/{contact_id}"
            update_data = {
                "properties": {
                    "churn_score": round(churn_probability * 100, 2),
                    "churn_risk_level": risk_level,
                    "last_churn_update": datetime.now().isoformat()
                }
            }

            response = requests.patch(update_url, headers=self.headers, json=update_data)
            response.raise_for_status()

            print(f"✓ Updated HubSpot contact {contact_id}: {risk_level} risk")
            return True

        except Exception as e:
            print(f"❌ HubSpot update failed: {str(e)}")
            return False

    def create_task(self, email: str, subject: str, notes: str) -> bool:
        """Create task/engagement for at-risk customer"""

        try:
            # Find contact
            search_url = f"{self.base_url}/crm/v3/objects/contacts/search"
            search_payload = {
                "filterGroups": [{
                    "filters": [{
                        "propertyName": "email",
                        "operator": "EQ",
                        "value": email
                    }]
                }]
            }

            response = requests.post(search_url, headers=self.headers, json=search_payload)
            response.raise_for_status()

            results = response.json().get('results', [])
            if not results:
                return False

            contact_id = results[0]['id']

            # Create task
            task_url = f"{self.base_url}/crm/v3/objects/tasks"
            task_data = {
                "properties": {
                    "hs_task_subject": subject,
                    "hs_task_body": notes,
                    "hs_task_status": "NOT_STARTED",
                    "hs_task_priority": "HIGH",
                    "hubspot_owner_id": results[0].get('properties', {}).get('hubspot_owner_id')
                },
                "associations": [{
                    "to": {"id": contact_id},
                    "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 204}]
                }]
            }

            response = requests.post(task_url, headers=self.headers, json=task_data)
            response.raise_for_status()

            print(f"✓ Created HubSpot task for contact {contact_id}")
            return True

        except Exception as e:
            print(f"❌ HubSpot task creation failed: {str(e)}")
            return False


# ============================================================================
# MICROSOFT DYNAMICS 365 INTEGRATION
# ============================================================================

class Dynamics365Integration:
    """
    Integrate with Microsoft Dynamics 365

    Features:
    - Update contact/account records
    - Create activities
    - Trigger Power Automate flows
    """

    def __init__(self, organization_url: str, access_token: str):
        """
        Initialize Dynamics 365 connection

        Args:
            organization_url: Your Dynamics 365 org URL (e.g., https://yourorg.crm.dynamics.com)
            access_token: OAuth access token
        """
        self.org_url = organization_url.rstrip('/')
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0"
        }

    def update_contact_churn_score(self, email: str, churn_probability: float, risk_level: str) -> bool:
        """Update custom churn fields on Dynamics 365 contact"""

        try:
            # Find contact
            filter_query = f"emailaddress1 eq '{email}'"
            url = f"{self.org_url}/api/data/v9.2/contacts?$filter={filter_query}&$select=contactid"

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            contacts = response.json().get('value', [])
            if not contacts:
                print(f"Contact not found: {email}")
                return False

            contact_id = contacts[0]['contactid']

            # Update churn score (assumes custom fields: new_churnscore, new_churnrisklevel)
            update_url = f"{self.org_url}/api/data/v9.2/contacts({contact_id})"
            update_data = {
                "new_churnscore": round(churn_probability * 100, 2),
                "new_churnrisklevel": risk_level,
                "new_lastchurnupdate": datetime.now().isoformat()
            }

            response = requests.patch(update_url, headers=self.headers, json=update_data)
            response.raise_for_status()

            print(f"✓ Updated Dynamics 365 contact: {risk_level} risk")
            return True

        except Exception as e:
            print(f"❌ Dynamics 365 update failed: {str(e)}")
            return False


# ============================================================================
# GENERIC WEBHOOK HANDLER
# ============================================================================

class WebhookHandler:
    """
    Generic webhook handler for any system

    Supports:
    - HMAC signature verification
    - Custom headers
    - Retry logic
    """

    @staticmethod
    def send_webhook(
        url: str,
        payload: Dict[str, Any],
        secret: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3
    ) -> bool:
        """
        Send webhook with optional HMAC signature

        Args:
            url: Webhook URL
            payload: JSON payload
            secret: Optional HMAC secret for signature
            custom_headers: Optional custom headers
            max_retries: Number of retry attempts

        Returns:
            bool: True if successful
        """

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AdventureWorks-Analytics/1.0"
        }

        if custom_headers:
            headers.update(custom_headers)

        # Add HMAC signature if secret provided
        if secret:
            payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
            signature = hmac.new(
                secret.encode('utf-8'),
                payload_bytes,
                hashlib.sha256
            ).hexdigest()
            headers['X-Hub-Signature-256'] = f"sha256={signature}"

        # Retry logic
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                response.raise_for_status()

                print(f"✓ Webhook sent successfully to {url}")
                return True

            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"❌ Webhook failed after {max_retries} attempts")
                    return False

        return False

    @staticmethod
    def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
        """
        Verify HMAC signature from incoming webhook

        Args:
            payload: Raw request body (bytes)
            signature: Signature from X-Hub-Signature-256 header
            secret: Your webhook secret

        Returns:
            bool: True if signature is valid
        """
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()

        # Remove "sha256=" prefix if present
        if signature.startswith('sha256='):
            signature = signature[7:]

        return hmac.compare_digest(expected_signature, signature)


# ============================================================================
# EXAMPLE WORKFLOWS
# ============================================================================

def workflow_1_salesforce_high_risk_customer():
    """Workflow: Update Salesforce when high-risk customer detected"""
    print("=" * 60)
    print("WORKFLOW 1: Salesforce High-Risk Customer Update")
    print("=" * 60)

    # Initialize Salesforce (replace with actual credentials)
    sf = SalesforceIntegration(
        instance_url="https://yourinstance.salesforce.com",
        access_token="your-access-token-here"
    )

    # Simulate high-risk customer from API
    customer_data = {
        "email": "john.doe@example.com",
        "churn_probability": 0.85,
        "risk_level": "High",
        "recommendation": "High-value customer at risk! Personal outreach recommended within 24 hours."
    }

    print(f"\nProcessing high-risk customer: {customer_data['email']}")
    print(f"Churn Risk: {customer_data['churn_probability']*100:.1f}%")

    # Update churn score in Salesforce
    sf.update_customer_churn_score(
        customer_data['email'],
        customer_data['churn_probability'],
        customer_data['risk_level']
    )

    # Create task for sales rep
    sf.create_retention_task(
        customer_data['email'],
        customer_data['recommendation']
    )

    print()


def workflow_2_hubspot_batch_update():
    """Workflow: Batch update HubSpot contacts with daily churn scores"""
    print("=" * 60)
    print("WORKFLOW 2: HubSpot Daily Batch Update")
    print("=" * 60)

    # Initialize HubSpot (replace with actual API key)
    hs = HubSpotIntegration(api_key="your-hubspot-api-key")

    # Simulate daily batch predictions
    daily_predictions = [
        {"email": "customer1@example.com", "churn_prob": 0.25, "risk": "Low"},
        {"email": "customer2@example.com", "churn_prob": 0.75, "risk": "High"},
        {"email": "customer3@example.com", "churn_prob": 0.50, "risk": "Medium"},
    ]

    print(f"\nUpdating {len(daily_predictions)} HubSpot contacts...")

    success_count = 0
    for pred in daily_predictions:
        if hs.update_contact_churn_score(pred['email'], pred['churn_prob'], pred['risk']):
            success_count += 1

            # Create task for high-risk customers
            if pred['risk'] == 'High':
                hs.create_task(
                    pred['email'],
                    "Urgent: High Churn Risk Customer",
                    f"Customer showing {pred['churn_prob']*100:.0f}% churn probability. Immediate action required."
                )

    print(f"\n✓ Updated {success_count}/{len(daily_predictions)} contacts")
    print()


def workflow_3_generic_webhook_to_custom_system():
    """Workflow: Send predictions to custom internal system"""
    print("=" * 60)
    print("WORKFLOW 3: Generic Webhook to Custom System")
    print("=" * 60)

    webhook_handler = WebhookHandler()

    # Prepare webhook payload
    payload = {
        "event": "high_churn_risk_detected",
        "timestamp": datetime.now().isoformat(),
        "customer_id": 12345,
        "email": "valued.customer@example.com",
        "churn_probability": 0.88,
        "risk_level": "High",
        "lifetime_value": 15000,
        "recommendation": "Assign to retention specialist immediately",
        "metadata": {
            "recency": 120,
            "frequency": 25,
            "last_purchase_date": "2024-06-15"
        }
    }

    print("\nSending webhook to internal system...")
    print(f"Event: {payload['event']}")
    print(f"Customer: {payload['email']} (Risk: {payload['churn_probability']*100:.0f}%)")

    # Send webhook with HMAC signature
    webhook_handler.send_webhook(
        url="https://your-internal-system.com/webhooks/analytics",
        payload=payload,
        secret="your-webhook-secret",  # For signature verification
        custom_headers={"X-Source": "AdventureWorks-Analytics"},
        max_retries=3
    )

    print()


def workflow_4_multi_crm_sync():
    """Workflow: Sync predictions to multiple CRM systems"""
    print("=" * 60)
    print("WORKFLOW 4: Multi-CRM Synchronization")
    print("=" * 60)

    customer = {
        "email": "important.client@example.com",
        "churn_probability": 0.92,
        "risk_level": "High"
    }

    print(f"\nSyncing {customer['email']} to multiple CRMs...")
    print(f"Risk: {customer['risk_level']} ({customer['churn_probability']*100:.1f}%)\n")

    # Salesforce
    print("1. Updating Salesforce...")
    # sf = SalesforceIntegration(...)
    # sf.update_customer_churn_score(customer['email'], customer['churn_probability'], customer['risk_level'])

    # HubSpot
    print("2. Updating HubSpot...")
    # hs = HubSpotIntegration(...)
    # hs.update_contact_churn_score(customer['email'], customer['churn_probability'], customer['risk_level'])

    # Dynamics 365
    print("3. Updating Dynamics 365...")
    # d365 = Dynamics365Integration(...)
    # d365.update_contact_churn_score(customer['email'], customer['churn_probability'], customer['risk_level'])

    print("\n✓ Customer synced across all CRM systems")
    print()


# ============================================================================
# FLASK WEBHOOK RECEIVER EXAMPLE
# ============================================================================

WEBHOOK_RECEIVER_EXAMPLE = """
# Example Flask app to receive webhooks from AdventureWorks Analytics API
# Save as: webhook_receiver.py

from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)
WEBHOOK_SECRET = "your-webhook-secret"

@app.route('/webhooks/churn', methods=['POST'])
def receive_churn_webhook():
    # Verify signature
    signature = request.headers.get('X-Hub-Signature-256', '')
    payload = request.get_data()

    if not verify_signature(payload, signature):
        return jsonify({"error": "Invalid signature"}), 403

    # Process webhook
    data = request.json
    customer_id = data.get('customer_id')
    churn_prob = data.get('churn_probability')
    risk_level = data.get('risk_level')

    print(f"Received churn alert: Customer {customer_id}, Risk: {risk_level}")

    # Your business logic here
    # - Update CRM
    # - Send email alert
    # - Create support ticket
    # - etc.

    return jsonify({"status": "received"}), 200

def verify_signature(payload, signature):
    expected = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()

    if signature.startswith('sha256='):
        signature = signature[7:]

    return hmac.compare_digest(expected, signature)

if __name__ == '__main__':
    app.run(port=5000)
"""

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """
    Run example workflows

    Note: These examples use placeholder credentials.
    Replace with actual CRM credentials to test.
    """

    print("\n" + "=" * 60)
    print("ADVENTUREWORKS ANALYTICS - CRM WEBHOOK INTEGRATION EXAMPLES")
    print("=" * 60 + "\n")

    print("Note: Examples use placeholder credentials")
    print("Update with actual CRM credentials to test\n")

    try:
        workflow_1_salesforce_high_risk_customer()
        workflow_2_hubspot_batch_update()
        workflow_3_generic_webhook_to_custom_system()
        workflow_4_multi_crm_sync()

        print("=" * 60)
        print("FLASK WEBHOOK RECEIVER EXAMPLE")
        print("=" * 60)
        print(WEBHOOK_RECEIVER_EXAMPLE)

        print("\n" + "=" * 60)
        print("✅ ALL WORKFLOW EXAMPLES COMPLETE")
        print("=" * 60 + "\n")

        print("Next steps:")
        print("1. Update CRM credentials in the code")
        print("2. Test with actual CRM sandbox environments")
        print("3. Deploy webhook receiver to production")
        print("4. Configure AdventureWorks API to send webhooks")
        print()

    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
