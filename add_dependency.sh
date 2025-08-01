#!/bin/bash

# Script to add dependencies to the correct requirements file

if [ $# -eq 0 ]; then
    echo "Usage: $0 <package_name> [version] [--heavy|--base]"
    echo ""
    echo "Examples:"
    echo "  $0 fastapi --base"
    echo "  $0 torch==2.0.1 --heavy"
    echo "  $0 numpy 1.24.3 --heavy"
    echo ""
    echo "If --heavy or --base is not specified, the script will prompt you."
    exit 1
fi

PACKAGE_NAME=$1
VERSION=$2
TYPE=$3

# If no type specified, prompt user
if [ -z "$TYPE" ]; then
    echo "Is this a heavy ML dependency (PyTorch, OpenCV, etc.) or a base application dependency?"
    echo "1) Heavy ML dependency (add to requirements_heavy.txt)"
    echo "2) Base application dependency (add to requirements_base.txt)"
    read -p "Enter your choice (1 or 2): " choice
    
    case $choice in
        1) TYPE="--heavy" ;;
        2) TYPE="--base" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
fi

# Construct the package line
if [ -n "$VERSION" ]; then
    PACKAGE_LINE="$PACKAGE_NAME==$VERSION"
else
    PACKAGE_LINE="$PACKAGE_NAME"
fi

# Add to appropriate file
case $TYPE in
    --heavy)
        echo "Adding $PACKAGE_LINE to requirements_heavy.txt"
        echo "$PACKAGE_LINE" >> portfolio-backend/requirements_heavy.txt
        echo "✅ Added to requirements_heavy.txt"
        ;;
    --base)
        echo "Adding $PACKAGE_LINE to requirements_base.txt"
        echo "$PACKAGE_LINE" >> portfolio-backend/requirements_base.txt
        echo "✅ Added to requirements_base.txt"
        ;;
    *)
        echo "Invalid type. Use --heavy or --base"
        exit 1
        ;;
esac

echo ""
echo "📝 Next steps:"
echo "1. Rebuild your Docker containers:"
echo "   docker-compose build --no-cache"
echo ""
echo "2. Restart your services:"
echo "   docker-compose up -d"
echo ""
echo "3. Verify the installation:"
echo "   docker-compose logs backend" 