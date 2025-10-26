#!/bin/bash
# Setup script for automated weekly reports
# Run this to schedule weekly executive summary reports

set -e  # Exit on error

echo "========================================"
echo "AdventureWorks Automated Reports Setup"
echo "========================================"
echo ""

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="$PROJECT_DIR/venv/bin/python"
SCRIPT_PATH="$PROJECT_DIR/dashboards/automated_reports.py"
LOG_PATH="$PROJECT_DIR/outputs/automated_reports.log"

echo "Configuration:"
echo "  Project: $PROJECT_DIR"
echo "  Python: $PYTHON_PATH"
echo "  Script: $SCRIPT_PATH"
echo "  Log: $LOG_PATH"
echo ""

# Verify files exist
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Error: Virtual environment not found at $PYTHON_PATH"
    echo "   Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: Script not found at $SCRIPT_PATH"
    exit 1
fi

# Test script
echo "Testing automated report generation..."
"$PYTHON_PATH" "$SCRIPT_PATH"

if [ $? -eq 0 ]; then
    echo "✅ Test successful!"
    echo ""
else
    echo "❌ Test failed. Please fix errors before scheduling."
    exit 1
fi

# Ask user for schedule
echo ""
echo "Select schedule for automated reports:"
echo "1) Every Monday at 9 AM (Recommended)"
echo "2) First day of month at 8 AM"
echo "3) Every weekday at 9 AM"
echo "4) Custom schedule"
echo "5) Skip scheduling (manual testing only)"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        CRON_SCHEDULE="0 9 * * 1"
        DESCRIPTION="Every Monday 9 AM"
        ;;
    2)
        CRON_SCHEDULE="0 8 1 * *"
        DESCRIPTION="First day of month 8 AM"
        ;;
    3)
        CRON_SCHEDULE="0 9 * * 1-5"
        DESCRIPTION="Weekdays 9 AM"
        ;;
    4)
        echo ""
        echo "Enter custom cron schedule (e.g., '0 9 * * 1' for Monday 9 AM):"
        echo "Format: minute hour day month weekday"
        echo "Help: https://crontab.guru/"
        read -p "Schedule: " CRON_SCHEDULE
        DESCRIPTION="Custom: $CRON_SCHEDULE"
        ;;
    5)
        echo ""
        echo "Skipping scheduling. You can run reports manually with:"
        echo "  $PYTHON_PATH $SCRIPT_PATH"
        echo ""
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Create cron job
CRON_COMMAND="$CRON_SCHEDULE $PYTHON_PATH $SCRIPT_PATH >> $LOG_PATH 2>&1"

echo ""
echo "Cron job to add:"
echo "  $CRON_COMMAND"
echo ""
echo "Description: $DESCRIPTION"
echo ""

read -p "Add this to crontab? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Cancelled. No changes made."
    exit 0
fi

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_COMMAND") | crontab -

echo ""
echo "✅ Cron job added successfully!"
echo ""
echo "Current crontab:"
crontab -l | grep -v "^#" | grep "$SCRIPT_PATH" || echo "  (no jobs found - this is unexpected)"
echo ""
echo "Next steps:"
echo "  1. Wait for scheduled time OR run manually: $PYTHON_PATH $SCRIPT_PATH"
echo "  2. Check logs: tail -f $LOG_PATH"
echo "  3. View reports: open outputs/automated_reports/"
echo ""
echo "To remove this cron job later:"
echo "  crontab -e  # Then delete the line containing '$SCRIPT_PATH'"
echo ""
echo "==============================================="
echo "Automated reports setup complete!"
echo "==============================================="
