"""
Email and Webhook Notification System
Phase 7: Real-Time API & Integration

Handles:
- Email alerts for high-risk customers/products
- Webhook notifications to external systems
- Scheduled digest reports
- Alert throttling to prevent spam

Author: AdventureWorks Data Science Team
Version: 1.0
Date: October 24, 2025
"""

import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from jinja2 import Template

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class EmailConfig:
    """Email configuration (load from environment variables in production)"""
    SMTP_SERVER = "smtp.gmail.com"  # Change to your SMTP server
    SMTP_PORT = 587
    SMTP_USERNAME = "your-email@adventureworks.com"  # Replace with actual email
    SMTP_PASSWORD = "your-app-password"  # Replace with app password
    FROM_EMAIL = "analytics@adventureworks.com"
    FROM_NAME = "AdventureWorks Analytics"

# Alert throttling: Don't send same alert more than once per hour
ALERT_CACHE = {}
THROTTLE_MINUTES = 60

# ============================================================================
# EMAIL TEMPLATES
# ============================================================================

CHURN_ALERT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #d32f2f; color: white; padding: 20px; text-align: center; }
        .content { background: #f5f5f5; padding: 20px; margin: 20px 0; }
        .alert-high { background: #ffebee; border-left: 4px solid #d32f2f; padding: 15px; }
        .alert-medium { background: #fff3e0; border-left: 4px solid #f57c00; padding: 15px; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat-box { text-align: center; padding: 15px; background: white; border-radius: 5px; }
        .stat-value { font-size: 32px; font-weight: bold; color: #1976d2; }
        .stat-label { font-size: 14px; color: #666; }
        .action-btn { display: inline-block; padding: 12px 30px; background: #1976d2; color: white; text-decoration: none; border-radius: 5px; margin: 10px 0; }
        .footer { text-align: center; color: #666; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚠️ High-Risk Customer Alert</h1>
        </div>

        <div class="content">
            <div class="alert-{{ risk_level|lower }}">
                <h2>{{ alert_title }}</h2>
                <p><strong>Customer ID:</strong> {{ customer_id }}</p>
                <p><strong>Churn Probability:</strong> {{ churn_probability }}%</p>
                <p><strong>Risk Level:</strong> {{ risk_level }}</p>
            </div>

            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{{ recency }}</div>
                    <div class="stat-label">Days Since Last Purchase</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ frequency }}</div>
                    <div class="stat-label">Total Orders</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">${{ monetary }}</div>
                    <div class="stat-label">Lifetime Value</div>
                </div>
            </div>

            <h3>Recommended Action:</h3>
            <p>{{ recommendation }}</p>

            <a href="{{ dashboard_url }}" class="action-btn">View in Dashboard</a>
        </div>

        <div class="footer">
            <p>This is an automated alert from AdventureWorks Analytics</p>
            <p>Generated at {{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
"""

RETURN_RISK_ALERT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #f57c00; color: white; padding: 20px; text-align: center; }
        .content { background: #f5f5f5; padding: 20px; margin: 20px 0; }
        .alert-box { background: #fff3e0; border-left: 4px solid #f57c00; padding: 15px; margin: 15px 0; }
        .product-info { background: white; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .action-btn { display: inline-block; padding: 12px 30px; background: #f57c00; color: white; text-decoration: none; border-radius: 5px; margin: 10px 0; }
        .footer { text-align: center; color: #666; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📦 High Return Risk Alert</h1>
        </div>

        <div class="content">
            <div class="alert-box">
                <h2>{{ alert_title }}</h2>
                <p><strong>Product:</strong> {{ product_name }} (ID: {{ product_id }})</p>
                <p><strong>Return Risk:</strong> {{ risk_probability }}%</p>
                <p><strong>Category:</strong> {{ category }}</p>
            </div>

            <div class="product-info">
                <h3>Product Details:</h3>
                <p><strong>Subcategory:</strong> {{ subcategory }}</p>
                <p><strong>List Price:</strong> ${{ list_price }}</p>
                <p><strong>Current Return Rate:</strong> {{ current_return_rate }}%</p>
            </div>

            <h3>Recommended Action:</h3>
            <p>{{ recommendation }}</p>

            <a href="{{ dashboard_url }}" class="action-btn">View in Dashboard</a>
        </div>

        <div class="footer">
            <p>This is an automated alert from AdventureWorks Analytics</p>
            <p>Generated at {{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
"""

DAILY_DIGEST_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { background: #1976d2; color: white; padding: 20px; text-align: center; }
        .section { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
        .kpi-card { background: white; padding: 15px; text-align: center; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .kpi-value { font-size: 28px; font-weight: bold; color: #1976d2; }
        .kpi-label { font-size: 14px; color: #666; margin-top: 5px; }
        .alert-list { list-style: none; padding: 0; }
        .alert-item { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #d32f2f; }
        .footer { text-align: center; color: #666; font-size: 12px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Daily Analytics Digest</h1>
            <p>{{ date }}</p>
        </div>

        <div class="section">
            <h2>Key Metrics</h2>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value">{{ high_risk_customers }}</div>
                    <div class="kpi-label">High-Risk Customers</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{{ high_risk_products }}</div>
                    <div class="kpi-label">High-Risk Products</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{{ predictions_today }}</div>
                    <div class="kpi-label">Predictions Today</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>⚠️ Top Alerts (Last 24 Hours)</h2>
            <ul class="alert-list">
                {% for alert in alerts %}
                <li class="alert-item">
                    <strong>{{ alert.type }}</strong>: {{ alert.message }}
                    <br><small>{{ alert.timestamp }}</small>
                </li>
                {% endfor %}
            </ul>
        </div>

        <div class="footer">
            <p>AdventureWorks Analytics - Automated Daily Digest</p>
            <p>To unsubscribe or modify preferences, contact analytics@adventureworks.com</p>
        </div>
    </div>
</body>
</html>
"""

# ============================================================================
# EMAIL FUNCTIONS
# ============================================================================

def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    attachments: Optional[List[Path]] = None
) -> bool:
    """
    Send HTML email with optional attachments

    Args:
        to_email: Recipient email address
        subject: Email subject
        html_content: HTML email body
        attachments: Optional list of file paths to attach

    Returns:
        bool: True if sent successfully, False otherwise
    """
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = f"{EmailConfig.FROM_NAME} <{EmailConfig.FROM_EMAIL}>"
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach HTML content
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)

        # Attach files
        if attachments:
            for file_path in attachments:
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename= {file_path.name}'
                        )
                        msg.attach(part)

        # Send email
        with smtplib.SMTP(EmailConfig.SMTP_SERVER, EmailConfig.SMTP_PORT) as server:
            server.starttls()
            server.login(EmailConfig.SMTP_USERNAME, EmailConfig.SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"Email sent successfully to {to_email}: {subject}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {str(e)}")
        return False

def should_send_alert(alert_key: str) -> bool:
    """
    Check if alert should be sent (throttling)

    Args:
        alert_key: Unique identifier for alert (e.g., "churn_customer_12345")

    Returns:
        bool: True if alert should be sent, False if throttled
    """
    current_time = datetime.now()

    if alert_key in ALERT_CACHE:
        last_sent = ALERT_CACHE[alert_key]
        if current_time - last_sent < timedelta(minutes=THROTTLE_MINUTES):
            logger.info(f"Alert throttled: {alert_key} (sent {THROTTLE_MINUTES} min ago)")
            return False

    ALERT_CACHE[alert_key] = current_time
    return True

def send_churn_alert(
    to_email: str,
    customer_id: int,
    churn_probability: float,
    risk_level: str,
    recency: float,
    frequency: int,
    monetary: float,
    recommendation: str,
    dashboard_url: str = "http://localhost:8501"
) -> bool:
    """Send churn alert email"""

    # Throttle check
    alert_key = f"churn_customer_{customer_id}"
    if not should_send_alert(alert_key):
        return False

    # Render template
    template = Template(CHURN_ALERT_TEMPLATE)
    html_content = template.render(
        alert_title=f"Customer #{customer_id} - {risk_level} Churn Risk",
        customer_id=customer_id,
        churn_probability=round(churn_probability * 100, 1),
        risk_level=risk_level,
        recency=int(recency),
        frequency=frequency,
        monetary=f"{monetary:,.2f}",
        recommendation=recommendation,
        dashboard_url=dashboard_url,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    subject = f"⚠️ High Churn Risk Alert: Customer #{customer_id}"

    return send_email(to_email, subject, html_content)

def send_return_risk_alert(
    to_email: str,
    product_id: int,
    product_name: str,
    risk_probability: float,
    category: str,
    subcategory: str,
    list_price: float,
    current_return_rate: float,
    recommendation: str,
    dashboard_url: str = "http://localhost:8501"
) -> bool:
    """Send return risk alert email"""

    # Throttle check
    alert_key = f"return_product_{product_id}"
    if not should_send_alert(alert_key):
        return False

    # Render template
    template = Template(RETURN_RISK_ALERT_TEMPLATE)
    html_content = template.render(
        alert_title=f"High Return Risk: {product_name}",
        product_id=product_id,
        product_name=product_name,
        risk_probability=round(risk_probability * 100, 1),
        category=category,
        subcategory=subcategory,
        list_price=f"{list_price:,.2f}",
        current_return_rate=round(current_return_rate * 100, 2),
        recommendation=recommendation,
        dashboard_url=dashboard_url,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    subject = f"📦 Return Risk Alert: {product_name}"

    return send_email(to_email, subject, html_content)

def send_daily_digest(
    to_emails: List[str],
    high_risk_customers: int,
    high_risk_products: int,
    predictions_today: int,
    alerts: List[Dict[str, Any]]
) -> bool:
    """Send daily digest email"""

    # Render template
    template = Template(DAILY_DIGEST_TEMPLATE)
    html_content = template.render(
        date=datetime.now().strftime("%B %d, %Y"),
        high_risk_customers=high_risk_customers,
        high_risk_products=high_risk_products,
        predictions_today=predictions_today,
        alerts=alerts[:10]  # Top 10 alerts
    )

    subject = f"📊 Daily Analytics Digest - {datetime.now().strftime('%Y-%m-%d')}"

    success_count = 0
    for email in to_emails:
        if send_email(email, subject, html_content):
            success_count += 1

    logger.info(f"Daily digest sent to {success_count}/{len(to_emails)} recipients")
    return success_count > 0

# ============================================================================
# WEBHOOK FUNCTIONS
# ============================================================================

def send_webhook(
    webhook_url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None
) -> bool:
    """
    Send webhook notification

    Args:
        webhook_url: Target webhook URL
        payload: JSON payload to send
        headers: Optional HTTP headers

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        default_headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AdventureWorks-Analytics/1.0'
        }

        if headers:
            default_headers.update(headers)

        response = requests.post(
            webhook_url,
            json=payload,
            headers=default_headers,
            timeout=10
        )

        response.raise_for_status()
        logger.info(f"Webhook sent successfully to {webhook_url}")
        return True

    except Exception as e:
        logger.error(f"Failed to send webhook to {webhook_url}: {str(e)}")
        return False

def send_churn_webhook(webhook_url: str, churn_data: Dict[str, Any]) -> bool:
    """Send churn prediction to webhook (e.g., CRM system)"""

    payload = {
        "event": "churn_prediction",
        "timestamp": datetime.now().isoformat(),
        "data": churn_data
    }

    return send_webhook(webhook_url, payload)

def send_return_risk_webhook(webhook_url: str, return_data: Dict[str, Any]) -> bool:
    """Send return risk prediction to webhook (e.g., inventory system)"""

    payload = {
        "event": "return_risk_prediction",
        "timestamp": datetime.now().isoformat(),
        "data": return_data
    }

    return send_webhook(webhook_url, payload)

# ============================================================================
# SLACK INTEGRATION (BONUS)
# ============================================================================

def send_slack_notification(
    webhook_url: str,
    message: str,
    severity: str = "warning"
) -> bool:
    """
    Send notification to Slack channel

    Args:
        webhook_url: Slack webhook URL
        message: Message text
        severity: "info", "warning", or "error"

    Returns:
        bool: True if successful
    """

    color_map = {
        "info": "#36a64f",
        "warning": "#ff9800",
        "error": "#d32f2f"
    }

    icon_map = {
        "info": ":information_source:",
        "warning": ":warning:",
        "error": ":rotating_light:"
    }

    payload = {
        "attachments": [{
            "color": color_map.get(severity, "#36a64f"),
            "title": f"{icon_map.get(severity, ':bell:')} AdventureWorks Analytics Alert",
            "text": message,
            "footer": "AdventureWorks Analytics",
            "ts": int(datetime.now().timestamp())
        }]
    }

    return send_webhook(webhook_url, payload)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of notification functions

    To use:
    1. Update EmailConfig with your SMTP settings
    2. Run this script to test notifications
    """

    # Example: Send churn alert
    print("Testing churn alert email...")
    result = send_churn_alert(
        to_email="manager@adventureworks.com",
        customer_id=12345,
        churn_probability=0.85,
        risk_level="High",
        recency=120,
        frequency=15,
        monetary=5000,
        recommendation="High-value customer at risk! Personal outreach recommended within 24 hours.",
        dashboard_url="http://localhost:8501"
    )
    print(f"Churn alert sent: {result}")

    # Example: Send return risk alert
    print("\nTesting return risk alert email...")
    result = send_return_risk_alert(
        to_email="quality@adventureworks.com",
        product_id=680,
        product_name="Mountain-100 Silver, 44",
        risk_probability=0.92,
        category="Bikes",
        subcategory="Mountain Bikes",
        list_price=3399.99,
        current_return_rate=0.05,
        recommendation="High return risk detected. Schedule quality audit and review customer feedback.",
        dashboard_url="http://localhost:8501"
    )
    print(f"Return risk alert sent: {result}")

    # Example: Send daily digest
    print("\nTesting daily digest email...")
    result = send_daily_digest(
        to_emails=["executive@adventureworks.com"],
        high_risk_customers=127,
        high_risk_products=31,
        predictions_today=1543,
        alerts=[
            {"type": "Churn", "message": "Customer #12345 - 85% churn risk", "timestamp": "10:30 AM"},
            {"type": "Return Risk", "message": "Product #680 - High return probability", "timestamp": "11:45 AM"},
        ]
    )
    print(f"Daily digest sent: {result}")

    print("\n✅ Notification testing complete!")
    print("\nTo enable in production:")
    print("1. Update EmailConfig with your SMTP server details")
    print("2. Set environment variables for sensitive credentials")
    print("3. Configure alert subscriptions in database")
    print("4. Schedule daily digest with cron/Task Scheduler")
